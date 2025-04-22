"""A design for a GeoFleX experiment."""

import enum
import itertools
import logging
from typing import Any
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
  treatment: list[set[str]] = []
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
  primary_metric: Metric
  experiment_budget_candidates: list[ExperimentBudget] = [None]
  secondary_metrics: list[Metric] = []
  alternative_hypothesis: str = "two-sided"
  alpha: float = 0.1
  eligible_methodologies: list[str] = ("TBR_MM", "TBR", "TM", "GBR")
  runtime_weeks_candidates: list[int] = [4, 6]
  n_cells: int = 2
  cell_volume_constraint_candidates: list[CellVolumeConstraint] = [None]
  geo_eligibility_candidates: list[GeoEligibility] = [None]
  random_seeds: list[int] = [0]
  effect_scope: EffectScope = EffectScope.ALL_GEOS

  model_config = pydantic.ConfigDict(extra="forbid")

  @pydantic.model_validator(mode="before")
  @classmethod
  def cast_none_cell_volume_constraint_candidates_to_unconstrained(
      cls, values: dict[str, Any]
  ) -> dict[str, Any]:
    """Casts None cell volume constraint candidates to unconstrained."""
    cell_volume_constraint_candidates = values.get(
        "cell_volume_constraint_candidates", [None]
    )
    n_cells = values.get("n_cells", 2)  # Default to 2 cells if not set.
    new_cell_volume_constraint_candidates = []
    for cell_volume_constraint in cell_volume_constraint_candidates:
      if cell_volume_constraint is None:
        new_cell_volume_constraint_candidates.append(
            CellVolumeConstraint(
                values=[None]*n_cells,
                constraint_type=CellVolumeConstraintType.NUMBER_OF_GEOS
            )
        )
      else:
        new_cell_volume_constraint_candidates.append(cell_volume_constraint)

    values["cell_volume_constraint_candidates"] = (
        new_cell_volume_constraint_candidates
    )
    return values

  @pydantic.field_validator("experiment_budget_candidates", mode="before")
  @classmethod
  def cast_none_budget_candidates_to_zero_spend(
      cls, experiment_budget_candidates: list[ExperimentBudget | None]
  ) -> list[ExperimentBudget]:
    """Casts None budget candidates to zero spend."""
    new_budget_candidates = []
    for budget_candidate in experiment_budget_candidates:
      if budget_candidate is None:
        new_budget_candidates.append(
            ExperimentBudget(
                value=0, budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE
            )
        )
      else:
        new_budget_candidates.append(budget_candidate)
    return new_budget_candidates

  @pydantic.model_validator(mode="after")
  def validate_budget_candidates_match_metrics(
      self,
  ) -> "ExperimentDesignSpec":
    """Ensures that the budget candidates are consistent with the metrics.

    1. If none of the metrics have a cost then budget doesn't matter.

    The budget only matters when looking at metrics like ROAS and CPA. For
    other metrics the budget doesn't matter for the purposes of the experiment
    design and analysis.

    In this case, if the user has specified multiple budget candidates but
    no cost metrics, then we warn the user and select the first budget candidate
    as the budget that will be used for the experiment design and analysis.

    2. If any of the metrics have a cost then all budget candidates must be
    non-zero.

    If any of the metrics have a cost then all budget candidates must be
    non-zero. If any of the budget candidates are zero then we raise an error.

    Returns:
      The values with the budget candidates possibly modified.
    """
    all_metrics = [self.primary_metric] + self.secondary_metrics
    has_cost_metric = any(
        metric.cost_per_metric or metric.metric_per_cost
        for metric in all_metrics
    )
    if not has_cost_metric:
      if len(self.experiment_budget_candidates) > 1:
        logger.warning(
            "None of the metrics have a cost, but there are multiple budget"
            " candidates. Dropping all but the first budget candidate, since"
            " the budget will have no influence on the design or the analysis"
            " results without any cost metrics."
        )
      self.experiment_budget_candidates = self.experiment_budget_candidates[:1]
    elif has_cost_metric:
      for budget_candidate in self.experiment_budget_candidates:
        if budget_candidate.value == 0.0:
          error_message = (
              "The experiment has cost metrics, but one of the budget"
              " candidates is zero. The cost metrics can only be used for a"
              " non-zero budget."
          )
          logger.error(error_message)
          raise ValueError(error_message)
    return self

  @pydantic.model_validator(mode="after")
  def check_experiment_budget_candidates_are_consistent_with_experiment_type(
      self,
  ) -> "ExperimentDesignSpec":
    for budget in self.experiment_budget_candidates:
      # Go dark experiment budgets must be a negative percentage change.
      if self.experiment_type == ExperimentType.GO_DARK:
        if budget.value >= 0:
          error_message = (
              "The percentage change budget must be negative for a go-dark"
              " experiment."
          )
          logger.error(error_message)
          raise ValueError(error_message)
        if budget.budget_type != ExperimentBudgetType.PERCENTAGE_CHANGE:
          error_message = (
              "The budget type must be 'percentage_change' for a go-dark"
              " experiment."
          )
          logger.error(error_message)
          raise ValueError(error_message)

      # Heavy-up and hold-back experiment budgets must be positive.
      if self.experiment_type in [
          ExperimentType.HEAVY_UP,
          ExperimentType.HOLD_BACK,
      ]:
        if budget.value <= 0:
          error_message = (
              "The daily budget must be positive for a heavy-up or hold-back"
              " experiment."
          )
          logger.error(error_message)
          raise ValueError(error_message)

      # Hold-back experiment budgets cannot be a percentage change.
      if self.experiment_type == ExperimentType.HOLD_BACK:
        if budget.budget_type == ExperimentBudgetType.PERCENTAGE_CHANGE:
          error_message = (
              "The budget type cannot be 'percentage_change' for a hold-back"
              " experiment."
          )
          logger.error(error_message)
          raise ValueError(error_message)

      # AB test experiment budgets must be zero.
      if self.experiment_type == ExperimentType.AB_TEST:
        if budget.value != 0:
          error_message = "The budget must be zero for an A/B test experiment."
          logger.error(error_message)
          raise ValueError(error_message)
    return self

  @pydantic.field_validator("alpha", mode="after")
  @classmethod
  def check_alpha_is_between_0_and_1(cls, alpha: float) -> float:
    if alpha < 0 or alpha > 1:
      error_message = "alpha must be between 0 and 1"
      logger.error(error_message)
      raise ValueError(error_message)
    return alpha

  @pydantic.field_validator("alternative_hypothesis", mode="after")
  @classmethod
  def check_alternative_hypothesis_is_valid(
      cls, alternative_hypothesis: str
  ) -> str:
    if alternative_hypothesis not in ["two-sided", "greater", "less"]:
      error_message = (
          "alternative_hypothesis must be one of 'two-sided', 'greater', or"
          " 'less'"
      )
      logger.error(error_message)
      raise ValueError(error_message)
    return alternative_hypothesis

  @pydantic.field_validator("primary_metric", mode="before")
  @classmethod
  def cast_primary_metric(cls, metric: Metric | str) -> Metric:
    if isinstance(metric, str):
      return Metric(name=metric)
    return metric

  @pydantic.field_validator("secondary_metrics", mode="before")
  @classmethod
  def cast_secondary_metrics(cls, metrics: list[Metric | str]) -> list[Metric]:
    return [
        Metric(name=metric) if isinstance(metric, str) else metric
        for metric in metrics
    ]

  @pydantic.model_validator(mode="before")
  @classmethod
  def cast_geo_eligibility_candidates(
      cls, values: dict[str, Any]
  ) -> dict[str, Any]:
    raw_geo_eligibility_candidates = values.get("geo_eligibility_candidates")
    n_cells = values.get("n_cells", 2)  # Default to 2 cells if not set.
    n_treatment_groups = n_cells - 1

    unconstrained_fixed_geos = GeoEligibility(
        control=[],
        treatment=[set()] * n_treatment_groups,
        exclude=[],
    )

    if raw_geo_eligibility_candidates is None:
      values["geo_eligibility_candidates"] = [
          unconstrained_fixed_geos.model_copy()
      ]
    else:
      values["geo_eligibility_candidates"] = [
          fixed_geos
          if fixed_geos is not None
          else unconstrained_fixed_geos.model_copy()
          for fixed_geos in raw_geo_eligibility_candidates
      ]
    return values

  @pydantic.field_validator("random_seeds", mode="after")
  @classmethod
  def check_at_least_one_random_seed(cls, random_seeds: list[int]) -> list[int]:
    if not random_seeds:
      error_message = "At least one random seed must be provided."
      logger.error(error_message)
      raise ValueError(error_message)
    return random_seeds

  @pydantic.model_validator(mode="after")
  def check_runtime_weeks_candidates_is_not_empty(
      self,
  ) -> "ExperimentDesignConstraints":
    """Checks that the runtime week candidates is not empty.

    Raises:
      ValueError: If runtime_weeks_candidates is empty.

    Returns:
      The ExperimentDesignConstraints object.
    """
    if not self.runtime_weeks_candidates:
      error_message = "runtime_weeks_candidates must be non-empty."
      logger.error(error_message)
      raise ValueError(error_message)
    return self

  @pydantic.model_validator(mode="after")
  def check_n_cells_greater_than_1(self) -> "ExperimentDesignConstraints":
    """Checks that n_cells is greater than or equal to 2.

    Raises:
        ValueError: If n_cells is less than 2.

    Returns:
        The ExperimentDesignConstraints object.
    """
    if self.n_cells < 2:
      error_message = "n_cells must be greater than or equal to 2"
      logger.error(error_message)
      raise ValueError(error_message)
    return self

  @pydantic.model_validator(mode="after")
  def check_cell_volume_constraint_candidates_match_n_cells(
      self,
  ) -> "ExperimentDesignConstraints":
    """Checks if cell volume constraint candidates matches number of cells.

    Raises:
        ValueError: If the number of cell volume constraints does not match
        the number of cells.

    Returns:
        The ExperimentDesignConstraints object.
    """
    for candidate in self.cell_volume_constraint_candidates:
      if candidate is not None:
        if len(candidate.values) != self.n_cells:
          error_message = (
              "The number of cell volume constraint values does not match the"
              " number of cells."
          )
          logger.error(error_message)
          raise ValueError(error_message)
    return self

  @pydantic.model_validator(mode="after")
  def check_geo_eligibility_matches_n_cells(
      self
  ) -> "ExperimentDesignConstraints":
    """Checks if number of treamtment arms matches n_cells."""
    for geo_eligibility in self.geo_eligibility_candidates:
      num_treatment_arms_in_geoeligibility = len(geo_eligibility.treatment)
      expected_num_treatment_arms = self.n_cells - 1

      if num_treatment_arms_in_geoeligibility != expected_num_treatment_arms:
        error_message = (
            "The number of treatment arms in the geo eligibility does not match"
            " the number of cells."
        )
        logger.error(error_message)
        raise ValueError(error_message)
    return self

  @pydantic.model_validator(mode="after")
  def check_metric_names_do_not_overlap(
      self,
  ) -> "ExperimentDesignSpec":
    all_metrics = [self.primary_metric] + self.secondary_metrics
    metric_names = [metric.name for metric in all_metrics]
    if len(metric_names) != len(set(metric_names)):
      error_message = "Metric names must be unique."
      logger.error(error_message)
      raise ValueError(error_message)
    return self

  @pydantic.model_validator(mode="after")
  def validate_that_volume_constraints_match_geo_eligibility_candidates(
      self,
  ) -> "ExperimentDesignSpec":
    """Check that the geo eligibility works with the volume constraints."""
    for geo_eligibility in self.geo_eligibility_candidates:
      for cell_volume_constraint in self.cell_volume_constraint_candidates:
        if cell_volume_constraint.constraint_type != (
            CellVolumeConstraintType.NUMBER_OF_GEOS
        ):
          continue

        if cell_volume_constraint.flexible:
          return self

        group_eligibility = [
            geo_eligibility.control
        ] + geo_eligibility.treatment
        for i in range(len(group_eligibility)):
          if cell_volume_constraint.values[i] is None:
            continue

          group = group_eligibility[i].copy()
          group -= geo_eligibility.exclude
          for j in range(len(group_eligibility)):
            if j == i:
              continue
            group -= group_eligibility[j]

          if len(group) > cell_volume_constraint.values[i]:
            error_message = (
                "The geo eligibility does not work with the cell volume"
                " constraint. The cell volume constraint requires"
                f" {cell_volume_constraint.values[i]} geos in cell {i}, but the"
                f" geo eligibility specifies {len(group)} geos that must be in"
                " that cell."
            )
            logger.error(error_message)
            raise ValueError(error_message)

    return self


class ExperimentDesignEvaluation(pydantic.BaseModel):
  """The evaluation results of an experiment design."""

  design_id: str
  minimum_detectable_effects: dict[Metric, float]  # One per metric
  false_positive_rates: dict[Metric, float]  # One per metric
  power_at_minimum_detectable_effect: dict[Metric, float]  # One per metric

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
  primary_metric: Metric
  experiment_budget: ExperimentBudget = None
  secondary_metrics: list[Metric] = []
  methodology: str
  runtime_weeks: int
  n_cells: int = 2
  alpha: float = 0.1
  alternative_hypothesis: str = "two-sided"
  geo_eligibility: GeoEligibility = None
  methodology_parameters: dict[str, Any] = {}
  random_seed: int = 0
  effect_scope: EffectScope = EffectScope.ALL_GEOS
  cell_volume_constraint: CellVolumeConstraint = None

  # The design id is autogenerated for a new design.
  design_id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))

  # Store geo assignment directly
  geo_assignment: GeoAssignment | None = None

  model_config = pydantic.ConfigDict(extra="forbid")

  @property
  def pretest_weeks(self) -> int:
    """The number of weeks in the pretest period.

      If a pretest period is to be used, this should be set as the pretest_weeks
      parameter in the methodology_parameters. If this parameter is not set then
      no pretest period is used.
    """
    return self.methodology_parameters.get("pretest_weeks", 0)

  @pydantic.field_validator("primary_metric", mode="before")
  @classmethod
  def cast_primary_metric(cls, metric: Metric | str) -> Metric:
    if isinstance(metric, str):
      return Metric(name=metric)
    return metric

  @pydantic.field_validator("secondary_metrics", mode="before")
  @classmethod
  def cast_secondary_metrics(cls, metrics: list[Metric | str]) -> list[Metric]:
    return [
        Metric(name=metric) if isinstance(metric, str) else metric
        for metric in metrics
    ]

  @pydantic.model_validator(mode="after")
  def check_cell_volume_constraint_per_group_matches_n_cells(
      self,
  ) -> "ExperimentDesign":
    """Checks that the cell volume constraint matches the number of cells."""
    if self.cell_volume_constraint is not None:
      if len(self.cell_volume_constraint.values) != self.n_cells:
        error_message = (
            "Length of cell_volume_constraint"
            f" ({len(self.cell_volume_constraint.values)}) does not match"
            f" n_cells ({self.n_cells})."
        )
        logger.error(error_message)
        raise ValueError(error_message)
    return self

  @pydantic.model_validator(mode="after")
  def check_geoassignment_matches_n_cells(self) -> "ExperimentDesign":
    """Checks if number of treatment arms in geoassignment matches n_cells."""
    if self.geo_assignment is not None:
      num_treatment_arms_in_assignment = len(
          self.geo_assignment.treatment
      )
      expected_num_treatment_arms = self.n_cells - 1
      if num_treatment_arms_in_assignment != expected_num_treatment_arms:
        error_message = (
            f"Expected {expected_num_treatment_arms} treatment arms "
            f"in geo_assignment (based on n_cells={self.n_cells}), "
            f"but found {num_treatment_arms_in_assignment}."
        )
        logger.error(error_message)
        raise ValueError(error_message)
    return self

  @pydantic.model_validator(mode="before")
  @classmethod
  def cast_geo_eligibility(cls, values: dict[str, Any]) -> dict[str, Any]:
    raw_geo_eligibility = values.get("geo_eligibility")
    n_cells = values.get("n_cells", 2)  # Default to 2 cells if not set.
    n_treatment_groups = n_cells - 1

    if raw_geo_eligibility is None:
      values["geo_eligibility"] = GeoEligibility(
          control=[],
          treatment=[set()] * n_treatment_groups,
          exclude=[],
      )
    return values

  @pydantic.model_validator(mode="after")
  def check_metric_names_do_not_overlap(
      self,
  ) -> "ExperimentDesign":
    all_metrics = [self.primary_metric] + self.secondary_metrics
    metric_names = [metric.name for metric in all_metrics]
    if len(metric_names) != len(set(metric_names)):
      error_message = "Metric names must be unique."
      logger.error(error_message)
      raise ValueError(error_message)
    return self

  @pydantic.model_validator(mode="after")
  def check_experiment_budget_is_valid(
      self,
  ) -> "ExperimentDesign":
    # Go dark experiment budgets must be a negative percentage change.
    if self.experiment_type == ExperimentType.GO_DARK:
      if self.experiment_budget.value >= 0:
        error_message = (
            "The percentage change budget must be negative for a go-dark"
            " experiment."
        )
        logger.error(error_message)
        raise ValueError(error_message)
      if (
          self.experiment_budget.budget_type
          != ExperimentBudgetType.PERCENTAGE_CHANGE
      ):
        error_message = (
            "The budget type must be 'percentage_change' for a go-dark"
            " experiment."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Heavy-up and hold-back experiment budgets must be positive.
    if self.experiment_type in [
        ExperimentType.HEAVY_UP,
        ExperimentType.HOLD_BACK,
    ]:
      if self.experiment_budget.value <= 0:
        error_message = (
            "The daily budget must be positive for a heavy-up or hold-back"
            " experiment."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Hold-back experiment budgets cannot be a percentage change.
    if self.experiment_type == ExperimentType.HOLD_BACK:
      if (
          self.experiment_budget.budget_type
          == ExperimentBudgetType.PERCENTAGE_CHANGE
      ):
        error_message = (
            "The budget type cannot be 'percentage_change' for a hold-back"
            " experiment."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # AB test experiment budgets must be zero.
    if self.experiment_type == ExperimentType.AB_TEST:
      if self.experiment_budget.value != 0:
        error_message = "The budget must be zero for an A/B test experiment."
        logger.error(error_message)
        raise ValueError(error_message)

    return self

  @pydantic.field_validator("experiment_budget", mode="before")
  @classmethod
  def cast_none_budget_to_zero_spend(
      cls, experiment_budget: ExperimentBudget | None
  ) -> ExperimentBudget:
    """Casts None budget to zero spend."""
    if experiment_budget is None:
      return ExperimentBudget(
          value=0, budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE
      )

    return experiment_budget

  @pydantic.model_validator(mode="after")
  def ensure_budget_is_consistent_with_metrics(
      self,
  ) -> "ExperimentDesignSpec":
    """If any of the metrics have a cost then the budget must be non-zero."""
    all_metrics = [self.primary_metric] + self.secondary_metrics
    has_cost_metric = any(
        metric.cost_per_metric or metric.metric_per_cost
        for metric in all_metrics
    )
    if has_cost_metric and self.experiment_budget.value == 0.0:
      error_message = (
          "The experiment has cost metrics, but one of the budget"
          " candidates is zero. The cost metrics can only be used for a"
          " non-zero budget."
      )
      logger.error(error_message)
      raise ValueError(error_message)
    return self

  @pydantic.model_validator(mode="before")
  @classmethod
  def cast_none_cell_volume_constraint_to_unconstrained(
      cls, values: dict[str, Any]
  ) -> dict[str, Any]:
    """Casts None cell volume constraint to unconstrained."""
    cell_volume_constraint = values.get("cell_volume_constraint")
    n_cells = values.get("n_cells", 2)  # Default to 2 cells if not set.
    if cell_volume_constraint is None:
      values["cell_volume_constraint"] = CellVolumeConstraint(
          values=[None] * n_cells,
          constraint_type=CellVolumeConstraintType.NUMBER_OF_GEOS,
      )

    return values
