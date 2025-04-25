"""A design for a GeoFleX experiment."""

import enum
import itertools
import logging
from typing import Annotated, Any
import uuid

import geoflex.metrics
import numpy as np
import pandas as pd
import pydantic


logger = logging.getLogger(__name__)
Metric = geoflex.metrics.Metric


class ExperimentType(enum.StrEnum):
  GO_DARK = "go_dark"  # Reduce spend in a campaign to measure incrementality.
  HEAVY_UP = (  # Increase spend in a campaign to measure incrementality.
      "heavy_up"
  )
  HOLD_BACK = "hold_back"  # Hold back a new campaign to measure incrementality.
  AB_TEST = "ab_test"  # Measure a campaign change without a change in spend.


class GeoEligibility(pydantic.BaseModel):
  """The geo eligibility for a geoflex experiment.

  Attributes:
    control: A set of geos eligible for the control group.
    treatment: A list of sets of geos eligible for the treatment group(s) where
      each set corresponds to a treatment arm.
    exclude: A set of geos eligible to be excluded from the experiment.
    all_geos: A list of all possible geos for the experiment.
    flexible: default is True. This means that geos not explicitly
      defined will be flexible in terms of assignment.
  """
  control: set[str] = set()
  treatment: list[set[str]] = [set()]
  exclude: set[str] = set()
  all_geos: set[str] = set()
  flexible: bool = True

  model_config = pydantic.ConfigDict(extra="forbid")

  @property
  def data(self) -> list[pd.DataFrame]:
    """Creates DataFrame indicating geo eligibility per treatment arm.

    Returns:
      A list of DataFrames, one per treatment cell.

      It is assumed that number of treatment arms is n_cells - 1. If the user
      has specified a 3 cell experiment, it is assumed there will be 2
      treatment arms and 1 control arm. Geoeligibility for control cell is
      captured in the treatment arm dataframe so a separate dataframe is not
      created for the control cell.

      If flexible is true, geos not explicitly defined as flexible will be
      assumed to be eligible for control or treatment in all cells.

      Each DataFrame has the following columns:
        geo: The geo id.
        control: 1 if the geo is eligible for the control, 0 otherwise.
          Value is consistent across all dataframes in the list.
        treatment: 1 if the geo is eligible for treatment in that arm, 0
          otherwise. If treatment arm eligibility is flexible, value is 1 for
          all dataframes in the list.
        exclude: 1 if the geo is eligible for exclusion, 0 otherwise.
          Value is consistent across all dataframes in the list.
    """
    num_treatment_arms = len(self.treatment)
    all_treatment_geos = set(itertools.chain.from_iterable(self.treatment))
    all_defined_geos = self.control | all_treatment_geos | self.exclude

    geos_to_process = sorted(list(self.all_geos | all_defined_geos))
    rows = [[] for _ in range(num_treatment_arms)]

    for geo in geos_to_process:
      # base eligibility is same in all treatment arms
      base_is_control = 1 if geo in self.control else 0
      base_is_exclude = 1 if geo in self.exclude else 0
      is_explicitly_defined = geo in all_defined_geos

      treatment_arm_eligibility = [
          1 if geo in arm_set else 0 for arm_set in self.treatment
      ]

      flexible_control = 0
      flexible_treatment = 0
      # if flexible is true, geos not explicitly defined are flexible
      if not is_explicitly_defined and self.flexible:
        flexible_control = 1
        flexible_treatment = 1

      # assign geos in each treatment arm(s)
      for arm_index in range(num_treatment_arms):
        row_data = {"geo": geo}
        row_data["control"] = base_is_control or flexible_control
        row_data["treatment"] = (
            treatment_arm_eligibility[arm_index]
            or flexible_treatment
        )
        row_data["exclude"] = base_is_exclude
        rows[arm_index].append(row_data)

    # create dataframes for each treatment arm
    dfs = []
    for arm_index in range(num_treatment_arms):
      df = pd.DataFrame(
          rows[arm_index] or [],
          columns=["geo", "control", "treatment", "exclude"]
      ).set_index("geo")
      dfs.append(df)

    return dfs


class GeoAssignment(GeoEligibility):
  """The geo assignment for the experiment."""

  @pydantic.model_validator(mode="after")
  def check_geos_not_in_multiple_groups(self) -> "GeoAssignment":
    """Checks that geos are not in multiple groups."""
    all_assignment_groups = [self.control, self.exclude] + self.treatment

    seen_geos = set()
    for group_set in all_assignment_groups:
      overlap = group_set & seen_geos
      if overlap:
        error_message = f"The following geos are in multiple groups: {overlap}"
        logger.error(error_message)
        raise ValueError(error_message)
      seen_geos.update(group_set)

    return self

  def make_geo_assignment_array(self, geos: list[str]) -> np.ndarray:
    """Creates an array with the geo assignment for a given list of geos.

    Args:
      geos: The list of geos to assign.

    Returns:
      An array with the geo assignment for the given list of geos. The array
      has the same length as the list of geos, and the value at index i
      represents the geo assignment for the geo at index i in the list of geos.
      The values are 0 for control, -1 for exclude, and 1+ for treatment, where
      1 represents the first cell, 2 represents the second cell, etc.
    """
    assignment = np.ones(len(geos)) * -1
    for i, geo in enumerate(geos):
      if geo in self.control:
        assignment[i] = 0
        continue

      for j, treatment_group in enumerate(self.treatment, 1):
        if geo in treatment_group:
          assignment[i] = j
          break
    return assignment.astype(int)


class ExperimentBudgetType(enum.StrEnum):
  PERCENTAGE_CHANGE = "percentage_change"
  DAILY_BUDGET = "daily_budget"
  TOTAL_BUDGET = "total_budget"


class ExperimentBudget(pydantic.BaseModel):
  """The budget for an experiment.

  The budget is the amount of money that can be spent on the experiment, and
  can be specified in different ways:

  - Budget type = percentage_change: The budget is a percentage change in the
      business as usual (BAU) spend. For example, a budget of 10% means that
      the experiment can spend 10% more than the BAU spend.
  - Budget type = daily_budget: The budget is a daily budget for the experiment.
      For example, a budget of $100,000 means that the experiment can spend
      $100,000 per day. Note that for a heavy-up experiment this is the
      incremental budget, meaning the increase on top of the BAU spend, not the
      total budget.
  - Budget type = total_budget: The budget is a total budget for the experiment.
      For example, a budget of $100,000 means that the experiment can spend
      $100,000 over the course of the experiment. Note that for a heavy-up
      experiment this is the incremental budget, meaning the increase on top of
      the BAU spend, not the total budget.

  For a Go-Dark experiment, the budget value should be negative, and is usually
  defined as a negative percentage change.

  For a Heavy-Up or Hold-Back experiment, the budget value should be positive,
  and is usually defined as a daily budget or a total budget.

  For a multi-cell test, the budget is the budget for each cell. For example, if
  the experiment has 3 cells and a budget of $100,000, then each treatment cell
  spend $100,000.

  Attributes:
    value: The value of the budget.
    budget_type: The type of the budget, one of "percentage_change",
      "daily_budget", or "total_budget".
  """

  value: float
  budget_type: ExperimentBudgetType

  model_config = pydantic.ConfigDict(extra="forbid")

  @pydantic.model_validator(mode="after")
  def check_percentage_change_is_not_below_minus_1(
      self,
  ) -> "ExperimentBudget":
    if (
        self.budget_type == ExperimentBudgetType.PERCENTAGE_CHANGE
        and self.value < -1.0
    ):
      error_message = (
          "Cannot have a percentage change budget below -1.0 (-100%)."
      )
      logger.error(error_message)
      raise ValueError(error_message)
    return self


class EffectScope(enum.StrEnum):
  """The scope of the effect to be measured in an experiment.

  ALL_GEOS: The goal of the experiment is to estimate the average effect
  of the treatment (ads) over all geos. In this case we require that the
  geos in the treatment group are representative of the whole population.

  TREATMENT_GEOS: The goal of the experiment is to estimate the average effect
  of the treatment (ads) over the geos in the treatment group. In this case
  the geos in the treatment group do not need to be representative of the whole
  population.

  For example, if you are testing the effectiveness of an existing ad campaign,
  then you want the effect scope to be ALL_GEOS, since you care about the
  effectiveness of your ads in all your geos. On the other hand, if you are
  testing the effectiveness of a new ad campaign that will only be shown in a
  specific subset of geos, then you want the effect scope to be TREATMENT_GEOS.
  """

  ALL_GEOS = "all_geos"
  TREATMENT_GEOS = "treatment_geos"


class CellVolumeConstraintType(enum.StrEnum):
  """The type of cell volume constraint to use in an experiment."""

  NUMBER_OF_GEOS = "number_of_geos"
  MAX_PERCENTAGE_OF_TOTAL_SPEND = "max_percentage_of_total_spend"
  MAX_PERCENTAGE_OF_TOTAL_RESPONSE = "max_percentage_of_total_response"


class CellVolumeConstraint(pydantic.BaseModel):
  """The cell volume constraint for an experiment.

  This can constrain either the number of geos per cell, the maximum percentage
  of total spend per cell, or the maximum percentage of total response per cell.

  Attributes:
    values: The values of the cell volume constraint. This should be a list
      of values, one for each cell. If the value is None, then there is no
      constraint on the cell volume for that cell.
    constraint_type: The type of the cell volume constraint, one of
      "number_of_geos", "max_percentage_of_total_spend", or
      "max_percentage_of_total_response".
  """

  values: list[float | int | None]
  constraint_type: CellVolumeConstraintType = (
      CellVolumeConstraintType.NUMBER_OF_GEOS
  )

  model_config = pydantic.ConfigDict(extra="forbid")

  @property
  def flexible(self) -> bool:
    """Returns True if the cell volume constraint is flexible."""
    return all(value is None for value in self.values)

  @pydantic.model_validator(mode="after")
  def check_constraint_type_is_implemented(
      self,
  ) -> "CellVolumeConstraint":
    if self.constraint_type != CellVolumeConstraintType.NUMBER_OF_GEOS:
      error_message = (
          "Only cell volume constraint of type 'number_of_geos' is currently"
          " implemented."
      )
      logger.error(error_message)
      raise NotImplementedError(error_message)
    return self


class ExperimentDesignEvaluation(pydantic.BaseModel):
  """The evaluation results of an experiment design."""

  design_id: str
  minimum_detectable_effects: dict[Metric, float]  # One per metric
  false_positive_rates: dict[Metric, float]  # One per metric
  power_at_minimum_detectable_effect: dict[Metric, float]  # One per metric

  model_config = pydantic.ConfigDict(extra="forbid")


@pydantic.BeforeValidator
def ensure_list(value: Any) -> Any:
  """Ensures that the value is a list."""
  if not isinstance(value, list):
    return [value]
  else:
    return value


@pydantic.BeforeValidator
def cast_string_to_metric(metric: str | Metric) -> Metric:
  """Converts a string to a metric."""
  if isinstance(metric, str):
    return Metric(name=metric)
  return metric


@pydantic.BeforeValidator
def fix_empty_geo_eligibility(
    geo_eligibility: GeoEligibility | dict[str, Any] | None,
    info: pydantic.ValidationInfo,
) -> GeoEligibility:
  """Fixes empty geo eligibility.

  If the geo eligibility is None, then it is set so that all geos are eligible
  for any arm of the experiment. If the treatment arms is an empty list, it
  makes sure it has the correct number of empty sets, one for each treatment
  arm.

  Args:
    geo_eligibility: The geo eligibility to fix.
    info: The validation info.

  Returns:
    The fixed geo eligibility.
  """
  n_cells = info.data.get("n_cells", 2)
  n_treatment_groups = n_cells - 1
  if geo_eligibility is None:
    return GeoEligibility(
        control=[],
        treatment=[set()] * n_treatment_groups,
        exclude=[],
    )

  if isinstance(geo_eligibility, dict):
    geo_eligibility = GeoEligibility.model_validate(geo_eligibility)

  all_treatment_geos = set().union(*geo_eligibility.treatment)
  if not all_treatment_geos:
    return geo_eligibility.model_copy(
        update=dict(treatment=[set()] * n_treatment_groups)
    )

  return geo_eligibility


@pydantic.BeforeValidator
def cast_none_budget_to_zero_spend(
    experiment_budget: ExperimentBudget | None,
) -> ExperimentBudget:
  """Casts None budget to zero spend."""
  if experiment_budget is None:
    return ExperimentBudget(
        value=0, budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE
    )

  return experiment_budget


@pydantic.BeforeValidator
def cast_none_cell_volume_constraint_to_unconstrained(
    cell_volume_constraint: CellVolumeConstraint | None,
    info: pydantic.ValidationInfo,
) -> dict[str, Any]:
  """Casts None cell volume constraint to unconstrained."""
  n_cells = info.data.get("n_cells", 2)
  if cell_volume_constraint is None:
    return CellVolumeConstraint(
        values=[None] * n_cells,
        constraint_type=CellVolumeConstraintType.NUMBER_OF_GEOS,
    )
  return cell_volume_constraint


@pydantic.BeforeValidator
def ensure_seed_is_list_and_not_empty(
    random_seeds: list[int] | None,
) -> list[int]:
  """Ensures that the random seeds is a list and not empty."""
  if random_seeds is None or not random_seeds:
    return [0]

  if isinstance(random_seeds, int):
    return [random_seeds]

  return random_seeds


@pydantic.AfterValidator
def check_n_cells_greater_than_1(n_cells: int) -> int:
  """Checks that n_cells is greater than or equal to 2."""
  if n_cells < 2:
    error_message = "n_cells must be greater than or equal to 2"
    logger.error(error_message)
    raise ValueError(error_message)
  return n_cells


@pydantic.AfterValidator
def check_alpha_is_between_0_and_1(alpha: float) -> float:
  """Checks that alpha is between 0 and 1."""
  if alpha < 0 or alpha > 1:
    error_message = "alpha must be between 0 and 1"
    logger.error(error_message)
    raise ValueError(error_message)
  return alpha


@pydantic.AfterValidator
def check_alternative_hypothesis_is_valid(alternative_hypothesis: str) -> str:
  """Checks that alternative_hypothesis is one of the valid options."""
  if alternative_hypothesis not in ["two-sided", "greater", "less"]:
    error_message = (
        "alternative_hypothesis must be one of 'two-sided', 'greater', or"
        " 'less'"
    )
    logger.error(error_message)
    raise ValueError(error_message)
  return alternative_hypothesis


@pydantic.AfterValidator
def check_cell_volume_constraint_matches_n_cells(
    cell_volume_constraint: CellVolumeConstraint, info: pydantic.ValidationInfo
) -> "CellVolumeConstraint":
  """Checks that the cell volume constraint matches the number of cells."""
  n_cells = info.data.get("n_cells", 2)
  if len(cell_volume_constraint.values) != n_cells:
    error_message = (
        "Length of cell_volume_constraint"
        f" ({len(cell_volume_constraint.values)}) does not match"
        f" n_cells ({n_cells})."
    )
    logger.error(error_message)
    raise ValueError(error_message)
  return cell_volume_constraint


@pydantic.AfterValidator
def check_geo_eligibility_matches_n_cells(
    geo_eligibility: GeoEligibility, info: pydantic.ValidationInfo
) -> "GeoEligibility":
  """Checks if number of treatment arms in geoeligibility matches n_cells."""
  n_cells = info.data.get("n_cells", 2)
  num_treatment_arms_in_eligibility = len(geo_eligibility.treatment)
  expected_num_treatment_arms = n_cells - 1
  if num_treatment_arms_in_eligibility != expected_num_treatment_arms:
    error_message = (
        f"Expected {expected_num_treatment_arms} treatment arms "
        f"in geo_eligibility (based on n_cells={n_cells}), "
        f"but found {num_treatment_arms_in_eligibility}."
    )
    logger.error(error_message)
    raise ValueError(error_message)
  return geo_eligibility


@pydantic.AfterValidator
def check_experiment_budget_is_valid(
    experiment_budget: ExperimentBudget, info: pydantic.ValidationInfo
) -> "ExperimentBudget":
  """Checks if the experiment budget is valid.

  Go dark experiments must have a negative percentage change budget.
  Heavy-up and hold-back experiments must have a positive budget.
  Hold-back experiments cannot have a percentage change budget.
  A/B test experiments must have a zero budget.

  Args:
    experiment_budget: The experiment budget to check.
    info: The validation info.

  Returns:
    The validated experiment budget.

  Raises:
    ValueError: If the experiment budget is not valid.
  """
  # Go dark experiment budgets must be a negative percentage change.
  experiment_type = info.data["experiment_type"]
  if experiment_type == ExperimentType.GO_DARK:
    if experiment_budget.value >= 0:
      error_message = (
          "The percentage change budget must be negative for a go-dark"
          " experiment."
      )
      logger.error(error_message)
      raise ValueError(error_message)
    if experiment_budget.budget_type != ExperimentBudgetType.PERCENTAGE_CHANGE:
      error_message = (
          "The budget type must be 'percentage_change' for a go-dark"
          " experiment."
      )
      logger.error(error_message)
      raise ValueError(error_message)

  # Heavy-up and hold-back experiment budgets must be positive.
  if experiment_type in [
      ExperimentType.HEAVY_UP,
      ExperimentType.HOLD_BACK,
  ]:
    if experiment_budget.value <= 0:
      error_message = (
          "The daily budget must be positive for a heavy-up or hold-back"
          " experiment."
      )
      logger.error(error_message)
      raise ValueError(error_message)

  # Hold-back experiment budgets cannot be a percentage change.
  if experiment_type == ExperimentType.HOLD_BACK:
    if experiment_budget.budget_type == ExperimentBudgetType.PERCENTAGE_CHANGE:
      error_message = (
          "The budget type cannot be 'percentage_change' for a hold-back"
          " experiment."
      )
      logger.error(error_message)
      raise ValueError(error_message)

  # AB test experiment budgets must be zero.
  if experiment_type == ExperimentType.AB_TEST:
    if experiment_budget.value != 0:
      error_message = "The budget must be zero for an A/B test experiment."
      logger.error(error_message)
      raise ValueError(error_message)

  return experiment_budget


@pydantic.AfterValidator
def check_budget_is_consistent_with_metrics(
    experiment_budget: ExperimentBudget,
    info: pydantic.ValidationInfo,
) -> "ExperimentBudget":
  """If any of the metrics have a cost then the budget must be non-zero."""
  all_metrics = [info.data["primary_metric"]] + info.data.get(
      "secondary_metrics", []
  )
  has_cost_metric = any(
      metric.cost_per_metric or metric.metric_per_cost for metric in all_metrics
  )
  if has_cost_metric and experiment_budget.value == 0.0:
    error_message = (
        "The experiment has cost metrics, but one of the budget"
        " candidates is zero. The cost metrics can only be used for a"
        " non-zero budget."
    )
    logger.error(error_message)
    raise ValueError(error_message)
  return experiment_budget


@pydantic.BeforeValidator
def drop_extra_budget_candidates_if_no_cost_metrics(
    experiment_budget_candidates: list[ExperimentBudget],
    info: pydantic.ValidationInfo,
) -> list[ExperimentBudget]:
  """If none of the metrics have a cost then budget doesn't matter.

  The budget only matters when looking at metrics like ROAS and CPA. For
  other metrics the budget doesn't matter for the purposes of the experiment
  design and analysis.

  In this case, if the user has specified multiple budget candidates but
  no cost metrics, then we warn the user and select the first budget candidate
  as the budget that will be used for the experiment design and analysis.

  Args:
    experiment_budget_candidates: The experiment budget candidates to validate.
    info: The validation info.

  Returns:
    The validated experiment budget candidates.
  """
  all_metrics = [info.data["primary_metric"]] + info.data.get(
      "secondary_metrics", []
  )
  has_cost_metric = any(
      metric.cost_per_metric or metric.metric_per_cost for metric in all_metrics
  )
  if not has_cost_metric and len(experiment_budget_candidates) > 1:
    logger.warning(
        "None of the metrics have a cost, but there are multiple budget"
        " candidates. Dropping all but the first budget candidate, since"
        " the budget will have no influence on the design or the analysis"
        " results without any cost metrics."
    )
    return experiment_budget_candidates[:1]

  return experiment_budget_candidates


@pydantic.AfterValidator
def check_secondary_metric_names_do_not_overlap(
    secondary_metrics: list[Metric], info: pydantic.ValidationInfo
) -> list[Metric]:
  """Checks that the secondary metric names are unique.

  They must not overlap with each other or the primary metric.

  Args:
    secondary_metrics: The secondary metrics to check.
    info: The validation info.

  Returns:
    The validated secondary metrics.
  """
  all_metrics = [info.data["primary_metric"]] + secondary_metrics
  metric_names = [metric.name for metric in all_metrics]
  if len(metric_names) != len(set(metric_names)):
    error_message = "Metric names must be unique."
    logger.error(error_message)
    raise ValueError(error_message)
  return secondary_metrics


@pydantic.AfterValidator
def check_runtime_weeks_candidates_not_empty(
    runtime_weeks_candidates: list[int],
) -> list[int]:
  """Checks that the runtime weeks candidates are not empty."""
  if not runtime_weeks_candidates:
    error_message = "Runtime weeks candidates must not be empty."
    logger.error(error_message)
    raise ValueError(error_message)
  return runtime_weeks_candidates


ValidatedMetric = Annotated[Metric, cast_string_to_metric]
ValidatedMetricList = Annotated[
    list[ValidatedMetric],
    ensure_list,
    check_secondary_metric_names_do_not_overlap,
]
ValidatedAlternativeHypothesis = Annotated[
    str, check_alternative_hypothesis_is_valid
]
ValidatedAlpha = Annotated[float, check_alpha_is_between_0_and_1]
ValidatedExperimentBudget = Annotated[
    ExperimentBudget,
    cast_none_budget_to_zero_spend,
    check_budget_is_consistent_with_metrics,
    check_experiment_budget_is_valid,
]
ValidatedExperimentBudgetCandidates = Annotated[
    list[ValidatedExperimentBudget],
    ensure_list,
    drop_extra_budget_candidates_if_no_cost_metrics,
]
ValidatedCellVolumeConstraint = Annotated[
    CellVolumeConstraint,
    cast_none_cell_volume_constraint_to_unconstrained,
    check_cell_volume_constraint_matches_n_cells,
]
ValidatedCellVolumeConstraintCandidates = Annotated[
    list[ValidatedCellVolumeConstraint], ensure_list
]
ValidatedGeoEligibility = Annotated[
    GeoEligibility,
    fix_empty_geo_eligibility,
    check_geo_eligibility_matches_n_cells,
]
ValidatedGeoEligibilityCandidates = Annotated[
    list[ValidatedGeoEligibility], ensure_list
]
ValidatedEligibleMethodologies = Annotated[list[str], ensure_list]
ValidatedRuntimeWeeksCandidates = Annotated[
    list[int], ensure_list, check_runtime_weeks_candidates_not_empty
]
ValidatedNCells = Annotated[int, check_n_cells_greater_than_1]
ValidatedSeedList = Annotated[list[int], ensure_seed_is_list_and_not_empty]


class ExperimentDesignSpec(pydantic.BaseModel):
  """All the inputs needed for geoflex to design an experiment.

  This includes some parameters of the experiment, such as the experiment type
  and the metrics. It also includes constraints on the design, such as the
  maximum and minimum number of weeks, the number of cells, and the number of
  geos per group.

  Attributes:
    experiment_type: The type of experiment to run.
    primary_metric: The primary response metric for the experiment. This is the
      metric that the experiment will be designed for.
    experiment_budget_candidates: The candidates for the experiment budget. The
      experiment design will choose the best configuration from this list. For a
      go-dark experiment, the budget value should be negative and is usually
      defined as a negative percentage change. For a heavy-up or hold-back
      experiment, the budget value should be positive and is usually defined as
      a daily budget or a total budget. For a heavy-up experiment, this is the
      incremental budget, meaning the increase on top of the BAU spend, not the
      total budget. If your metrics do not include cost, or you are running an
      A/B test, then you do not need to specify a budget.
    secondary_metrics: The secondary response metrics for the experiment. These
      are the metrics that the experiment will also measure, but are not as
      important as the primary metric.
    alternative_hypothesis: The alternative hypothesis for the experiment. Must
      be one of "two-sided", "greater", or "less". Defaults to "two-sided".
    alpha: The significance level for the experiment. Defaults to 0.1.
    eligible_methodologies: The eligible methodologies for the experiment.
      Defaults to all methodologies except RCT.
    runtime_weeks_candidates: The candidates for the number of weeks the
      experiment can run. The experiment design will choose the best
      configuration from this list.
    n_cells: The number of cells to use for the experiment. Must be at least 2.
    cell_volume_constraint_candidates: A list of CellVolumeConstraints.The
      experiment design will choose the best configuration from this list. Each
      constraint must have a value for each cell. If the constraint value is
      None, then there is no constraint on the cell volume for that cell.
    geo_eligibility_candidates: The geo eligibility candidates for the
      experiment.
    random_seeds: The random seeds to use for the experiment. If any random
      number generator is used in the geo assignment, then this seed will be
      used. This ensures that the geo assignment is reproducible. Setting
      multiple options for seeds lets you explore different random assignments.
    effect_scope: The scope of the effect to be measured in the experiment. This
      can be either "all_geos" or "treatment_geos". Defaults to "all_geos". See
      the EffectScope enum for more details.
  """

  experiment_type: ExperimentType
  primary_metric: ValidatedMetric
  n_cells: ValidatedNCells = 2
  secondary_metrics: ValidatedMetricList = []
  experiment_budget_candidates: ValidatedExperimentBudgetCandidates = (
      pydantic.Field(default=[None], validate_default=True)
  )
  alternative_hypothesis: ValidatedAlternativeHypothesis = "two-sided"
  alpha: ValidatedAlpha = 0.1
  eligible_methodologies: ValidatedEligibleMethodologies = [
      "TBR_MM",
      "TBR",
      "TM",
      "GBR",
  ]
  runtime_weeks_candidates: ValidatedRuntimeWeeksCandidates = [4]
  cell_volume_constraint_candidates: ValidatedCellVolumeConstraintCandidates = (
      pydantic.Field(default=[None], validate_default=True)
  )
  geo_eligibility_candidates: ValidatedGeoEligibilityCandidates = (
      pydantic.Field(default=[None], validate_default=True)
  )
  random_seeds: ValidatedSeedList = [0]
  effect_scope: EffectScope = pydantic.Field(
      default=EffectScope.ALL_GEOS, validate_default=True
  )

  model_config = pydantic.ConfigDict(extra="forbid")


class ExperimentDesign(pydantic.BaseModel):
  """An experiment design for a GeoFleX experiment.

  Attributes:
    experiment_type: The type of experiment to run.
    design_id: The unique identifier for the experiment design. This is
      autogenerated for a new design.
    primary_metric: The primary response metric for the experiment. This is the
      metric that the experiment will be designed to measure, and should be the
      main decision making metric.
    experiment_budget: The experiment budget for the experiment. This can be a
      percentage change, daily budget, or total budget. For a go-dark
      experiment, the budget value should be a negative percentage change. For a
      heavy-up or hold-back experiment, the budget value should be positive and
      is usually defined as a daily budget or a total budget. For a heavy-up
      experiment, this is the incremental budget, meaning the increase on top of
      the BAU spend, not the total budget. If your metrics do not include cost,
      or you are running an A/B test, then you do not need to specify a budget.
    secondary_metrics: The secondary response metrics for the experiment. These
      are the metrics that the experiment will also measure, but are not as
      important as the primary metric.
    methodology: The methodology to use for the experiment.
    methodology_parameters: The parameters specific to the methodology.
    runtime_weeks: The number of weeks to run the experiment.
    n_cells: The number of cells to use for the experiment. This must be at
      least 2.
    alpha: The significance level for the experiment.
    alternative_hypothesis: The alternative hypothesis for the experiment.
    geo_eligibility: The geo eligibility for the experiment. This is used to
      specify which geos are eligible to be in which groups in the experiment.
      If None, then all geos are eligible for all groups.
    cell_volume_constraint: The cell volume constraint for the experiment. This
      is used to specify the constraint on the number of geos per cell. If None,
      then there is no constraint.
    constraints: The constraints for the experiment.
    geo_assignment: The geo assignment for the experiment. This is set after the
      design is created, when the geos are assigned.
    random_seed: The random seed used for the experiment. This is used to ensure
      that the experiment is reproducible.
    effect_scope: The scope of the effect to be measured in the experiment. This
      can be either "all_geos" or "treatment_geos". Defaults to "all_geos". See
      the EffectScope enum for more details.
  """
  experiment_type: ExperimentType
  n_cells: ValidatedNCells = 2
  alpha: ValidatedAlpha = 0.1
  primary_metric: ValidatedMetric
  secondary_metrics: ValidatedMetricList = []
  experiment_budget: ValidatedExperimentBudget = pydantic.Field(
      default=None, validate_default=True
  )
  methodology: str
  runtime_weeks: int
  alternative_hypothesis: ValidatedAlternativeHypothesis = "two-sided"
  geo_eligibility: ValidatedGeoEligibility = pydantic.Field(
      default=None, validate_default=True
  )
  methodology_parameters: dict[str, Any] = {}
  random_seed: int = 0
  effect_scope: EffectScope = EffectScope.ALL_GEOS
  cell_volume_constraint: ValidatedCellVolumeConstraint = pydantic.Field(
      default=None, validate_default=True
  )

  # The design id is autogenerated for a new design.
  design_id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))

  # Store geo assignment directly
  geo_assignment: GeoAssignment | None = None

  model_config = pydantic.ConfigDict(extra="forbid")
