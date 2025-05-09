"""The module containing all the classes to define an experiment design."""

import enum
import itertools
import logging
from typing import Annotated, Any
import uuid
import geoflex.metrics
import numpy as np
import pandas as pd
import pydantic
import yaml


logger = logging.getLogger(__name__)
Metric = geoflex.metrics.Metric


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

  def to_dict(self) -> dict[str, Any]:
    """Creates a dict with the treatment groups as separate keys."""
    out = {
        "control": sorted(list(self.control)),
        "exclude": sorted(list(self.exclude)),
    }
    for i, treatment_cell in enumerate(self.treatment, 1):
      out[f"treatment_{i}"] = sorted(list(treatment_cell))

    out_filtered = {k: v for k, v in out.items() if v}
    return out_filtered


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

  def get_n_geos_per_group(self) -> pd.Series:
    """Creates three columns for the number of geos per group for printing."""
    out = {
        "n_geos_control": len(self.control),
        "n_geos_exclude": len(self.exclude),
    }
    for i, treatment_cell in enumerate(self.treatment):
      out[f"n_geos_treatment_{i}"] = len(treatment_cell)

    return pd.Series(out)


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
    value: The value of the budget. This can be either a single value or a list
      of values, one for each cell. If a single value is provided, then it will
      assume the same budget for all cells.
    budget_type: The type of the budget, one of "percentage_change",
      "daily_budget", or "total_budget".
  """

  value: float | list[float]
  budget_type: ExperimentBudgetType

  model_config = pydantic.ConfigDict(extra="forbid")

  @pydantic.model_validator(mode="after")
  def check_percentage_change_is_not_below_minus_1(
      self,
  ) -> "ExperimentBudget":
    values_list = self.value if isinstance(self.value, list) else [self.value]
    for value in values_list:
      if (
          self.budget_type == ExperimentBudgetType.PERCENTAGE_CHANGE
          and value < -1.0
      ):
        error_message = (
            "Cannot have a percentage change budget below -1.0 (-100%)."
        )
        logger.error(error_message)
        raise ValueError(error_message)
    return self

  def _single_value_to_string(self, value: float) -> str:
    if np.isclose(value, 0.0):
      return "No Budget"
    elif self.budget_type == ExperimentBudgetType.PERCENTAGE_CHANGE:
      return f"{value:.0%}"
    elif self.budget_type == ExperimentBudgetType.DAILY_BUDGET:
      return f"${value} per day"
    elif self.budget_type == ExperimentBudgetType.TOTAL_BUDGET:
      return f"${value} total"
    else:
      # Fallback, not needed now, in case we add more budget types in future.
      return f"Value = {value}, Type = {self.budget_type}"

  def __str__(self) -> str:
    if isinstance(self.value, list):
      return ", ".join([
          f"Cell {i}: {self._single_value_to_string(value)}"
          for i, value in enumerate(self.value, 1)
      ])
    else:
      final_string = self._single_value_to_string(self.value)
      if self.budget_type != ExperimentBudgetType.PERCENTAGE_CHANGE:
        final_string += " per cell"
      return final_string


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

  def __str__(self) -> str:
    """Returns the cell volume constraint as a string for printing."""
    if self.constraint_type == CellVolumeConstraintType.NUMBER_OF_GEOS:
      suffix = " geos"
    elif (
        self.constraint_type
        == CellVolumeConstraintType.MAX_PERCENTAGE_OF_TOTAL_SPEND
    ):
      suffix = "% of spend"
    elif (
        self.constraint_type
        == CellVolumeConstraintType.MAX_PERCENTAGE_OF_TOTAL_RESPONSE
    ):
      suffix = "% of response metric"
    else:
      # Fallback, not needed now, in case we add more constraint types.
      suffix = f" {self.constraint_type.value}"

    values = []
    for i, value in enumerate(self.values):
      suffix_i = suffix if value else ""
      if i == 0:
        values.append(f"control: {value}{suffix_i}")
      else:
        values.append(f"treatment_{i}: {value}{suffix_i}")

    return ", ".join(values)


class SingleEvaluationResult(pydantic.BaseModel):
  """A single set of evaluation results.

  The evaluation result of an single metric, in a single cell of one
  experiment design.

  Results for relative effects are None if the metric is a cost based metric
  like ROAS or CPA, where they don't make sense.

  Attributes:
    standard_error_absolute_effect: The standard error of the absolute effect.
    standard_error_relative_effect: The standard error of the relative effect.
    coverage_absolute_effect: The coverage of the absolute effect.
    coverage_relative_effect: The coverage of the relative effect.
    all_checks_pass: Whether all checks pass.
    failing_checks: The list of failing checks. If all checks pass, this will be
      an empty list.
  """

  standard_error_absolute_effect: float
  standard_error_relative_effect: float | None
  coverage_absolute_effect: float
  coverage_relative_effect: float | None
  all_checks_pass: bool
  failing_checks: list[str]

  model_config = pydantic.ConfigDict(extra="forbid")


class ExperimentDesignEvaluationResults(pydantic.BaseModel):
  """The evaluation results of an experiment design.

  Contains the evaluation results of an experiment design. This will have
  evaluation results for each metric, and each cell of the experiment design.

  Attributes:
    primary_metric_name: The name of the primary metric.
    alpha: The alpha value of the experiment.
    alternative_hypothesis: The alternative hypothesis of the experiment.
    all_metric_results_per_cell: A dictionary of metric name to a list of
      evaluation results for that metric and cell.
    primary_metric_results_per_cell: The evaluation results of the primary
      metric for each cell.
    all_metric_results: The evaluation results of all metrics.
    primary_metric_results: The evaluation results of the primary metric.
    representiveness_scores_per_cell: The representativeness score of each of
      the treatment group cells. If the effect scope of the design is not
      ALL_GEOS, then this will be 0 for all cells.
    is_valid_design: Whether the design is valid. This will be false if the
      methodology is not eligible for the design. All of the design results will
      be None in this case.
  """

  primary_metric_name: str
  alpha: float
  alternative_hypothesis: str
  all_metric_results_per_cell: (
      dict[str, list[SingleEvaluationResult]] | None
  ) = None
  representiveness_scores_per_cell: list[float] | None = None
  is_valid_design: bool

  model_config = pydantic.ConfigDict(extra="forbid")

  @property
  def primary_metric_results_per_cell(
      self,
  ) -> list[SingleEvaluationResult] | None:
    """Returns the evaluation result of the primary metric for each cell."""
    if self.all_metric_results is None:
      return None
    return self.all_metric_results_per_cell[self.primary_metric_name]

  @property
  def representiveness_score(self) -> float | None:
    """Returns the minimum representativeness score across all cells."""
    if self.representiveness_scores_per_cell is None:
      return None
    return min(self.representiveness_scores_per_cell)

  @property
  def all_metric_results(self) -> dict[str, SingleEvaluationResult] | None:
    """Returns the evaluation result of all metrics.

    The metrics are aggregated across all cells in a worst case scenario, taking
    the highest standard error, lowest coverage, and all checks must pass in
    all cells.
    """
    if self.all_metric_results_per_cell is None:
      return None

    results = {}
    for metric, results_per_cell in self.all_metric_results_per_cell.items():
      standard_error_absolute_effect = max(
          [result.standard_error_absolute_effect for result in results_per_cell]
      )
      coverage_absolute_effect = min(
          [result.coverage_absolute_effect for result in results_per_cell]
      )

      if results_per_cell[0].standard_error_relative_effect is not None:
        # Assume all the relative effects exist if the first one does.
        standard_error_relative_effect = max([
            result.standard_error_relative_effect for result in results_per_cell
        ])
        coverage_relative_effect = min(
            [result.coverage_relative_effect for result in results_per_cell]
        )
      else:
        standard_error_relative_effect = None
        coverage_relative_effect = None

      all_checks_pass = all(
          [result.all_checks_pass for result in results_per_cell]
      )
      failing_checks = list(
          set().union(*([result.failing_checks for result in results_per_cell]))
      )

      results[metric] = SingleEvaluationResult(
          standard_error_absolute_effect=standard_error_absolute_effect,
          standard_error_relative_effect=standard_error_relative_effect,
          coverage_absolute_effect=coverage_absolute_effect,
          coverage_relative_effect=coverage_relative_effect,
          all_checks_pass=all_checks_pass,
          failing_checks=failing_checks,
      )
    return results

  @property
  def primary_metric_results(self) -> SingleEvaluationResult | None:
    """Returns the evaluation result of the primary metric."""
    if self.all_metric_results is None:
      return None
    return self.all_metric_results[self.primary_metric_name]

  def get_mde(
      self,
      target_power: float,
      relative: bool,
      aggregate_across_cells: bool,
  ) -> dict[str, list[float] | float | None]:
    """Returns the MDE for each metric.

    If relative is True, then the relative effect MDE is returned. Otherwise,
    the absolute effect MDE is returned. If the relative MDE is not available
    then None is returned.

    Args:
      target_power: The target power.
      relative: Whether to use the relative effect MDE.
      aggregate_across_cells: Whether to aggregate the MDE across all cells.

    Returns:
      A dictionary of metric name to the MDE for that metric. If
      aggregate_across_cells is True, then the MDE for each metric will be a
      single float. Otherwise, the MDE for each cell will be a list of floats.
      If the relative MDE is not available, then None will be returned. If the
      design is invalid then this will be an empty dictionary.
    """
    if self.all_metric_results_per_cell is None:
      return {}

    mde_per_cell = {}
    for (
        metric_name,
        results_per_cell,
    ) in self.all_metric_results_per_cell.items():
      metric_was_inverted = "__INVERTED__" in metric_name
      if metric_was_inverted:
        metric_name = metric_name.replace(" __INVERTED__", "")

      mde_per_cell[metric_name] = []

      if (
          relative
          and results_per_cell[0].standard_error_relative_effect is None
      ):
        mde_per_cell[metric_name] = None
        continue

      for result in results_per_cell:
        if relative:
          standard_error = result.standard_error_relative_effect
        else:
          standard_error = result.standard_error_absolute_effect

        mde = geoflex.evaluation.calculate_minimum_detectable_effect_from_stats(
            standard_error=standard_error,
            alternative=self.alternative_hypothesis,
            power=target_power,
            alpha=self.alpha,
        )

        if metric_was_inverted:
          mde = 1.0 / mde

        mde_per_cell[metric_name].append(float(mde))

      if aggregate_across_cells:
        if metric_was_inverted:
          mde_per_cell[metric_name] = min(mde_per_cell[metric_name])
        else:
          mde_per_cell[metric_name] = max(mde_per_cell[metric_name])

    return mde_per_cell

  def get_summary_dict(
      self,
      target_power: float = 0.8,
      use_relative_effects_where_possible: bool = True,
  ) -> dict[str, Any]:
    """Returns the evaluation results as a summary pandas series."""
    if self.all_metric_results is None:
      return {
          "failing_checks": ["Design is not eligible for this methodology"],
          "all_checks_pass": False,
          "primary_metric_failing_checks": [
              "Design is not eligible for this methodology"
          ],
          "primary_metric_all_checks_pass": False,
      }

    absolute_mde = self.get_mde(
        target_power=target_power,
        relative=False,
        aggregate_across_cells=True,
    )
    if use_relative_effects_where_possible:
      relative_mde = self.get_mde(
          target_power=target_power,
          relative=True,
          aggregate_across_cells=True,
      )
    else:
      relative_mde = {}

    output = {
        "failing_checks": [],
        "all_checks_pass": True,
        "representiveness_score": self.representiveness_score,
    }
    for metric_name, result in self.all_metric_results.items():
      if "__INVERTED__" in metric_name:
        metric_name = metric_name.replace(" __INVERTED__", "")

      if (mde := relative_mde.get(metric_name)) is not None:
        standard_error = result.standard_error_relative_effect
        mde_name_prefix = "Relative "
      else:
        mde = absolute_mde[metric_name]
        standard_error = result.standard_error_absolute_effect
        mde_name_prefix = ""

      if metric_name == self.primary_metric_name:
        is_primary_string = ", primary metric"
        output["primary_metric_failing_checks"] = result.failing_checks
        output["primary_metric_all_checks_pass"] = result.all_checks_pass
        output["primary_metric_standard_error"] = standard_error
      else:
        is_primary_string = ""

      mde_name = f"{mde_name_prefix}MDE ({metric_name}{is_primary_string})"
      output[mde_name] = mde

      output["failing_checks"] += result.failing_checks
      output["all_checks_pass"] &= result.all_checks_pass

    return output


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
def ensure_list(value: Any) -> Any:
  """Ensures that the value is a list."""
  if not isinstance(value, list):
    return [value]
  else:
    return value


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

  budget_values = experiment_budget.value
  if not isinstance(budget_values, list):
    budget_values = [budget_values]

  for budget_value in budget_values:
    if has_cost_metric and budget_value == 0.0:
      error_message = (
          "The experiment has cost metrics, but one of the budget values is"
          " zero. The cost metrics can only be used with non-zero budgets in"
          " all treatment cells."
      )
      logger.error(error_message)
      raise ValueError(error_message)

  return experiment_budget


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
def check_budget_is_consistent_with_n_cells(
    budget: ExperimentBudget, info: pydantic.ValidationInfo
) -> list[Any]:
  """Checks that the number of budgets matches the number of treatment cells."""
  if not isinstance(budget.value, list):
    return budget

  n_cells = info.data.get("n_cells", 2)
  num_treatment_cells = n_cells - 1
  num_budget_values = len(budget.value)
  if num_budget_values != num_treatment_cells:
    error_message = (
        f"Length of list ({num_budget_values}) does not match"
        f" the number of treatment cells ({num_treatment_cells})."
    )
    logger.error(error_message)
    raise ValueError(error_message)

  return budget


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
    check_budget_is_consistent_with_n_cells,
    check_budget_is_consistent_with_metrics,
]
ValidatedCellVolumeConstraint = Annotated[
    CellVolumeConstraint,
    cast_none_cell_volume_constraint_to_unconstrained,
    check_cell_volume_constraint_matches_n_cells,
]
ValidatedGeoEligibility = Annotated[
    GeoEligibility,
    fix_empty_geo_eligibility,
    check_geo_eligibility_matches_n_cells,
]
ValidatedNCells = Annotated[int, check_n_cells_greater_than_1]


class ExperimentDesign(pydantic.BaseModel):
  """An experiment design for a GeoFleX experiment.

  Attributes:
    design_id: The unique identifier for the experiment design. This is
      autogenerated for a new design.
    primary_metric: The primary response metric for the experiment. This is the
      metric that the experiment will be designed to measure, and should be the
      main decision making metric.
    experiment_budget: The experiment budget for the experiment. This can be a
      percentage change, daily budget, or total budget. For a go-dark
      experiment, the budget value should be negative. For a
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
    evaluation_results: The evaluation results for the experiment design. This
      is set after the design is created, when the design is evaluated.
  """

  primary_metric: ValidatedMetric
  methodology: str
  runtime_weeks: int

  n_cells: ValidatedNCells = 2
  alpha: ValidatedAlpha = 0.1
  secondary_metrics: ValidatedMetricList = []
  experiment_budget: ValidatedExperimentBudget = pydantic.Field(
      default=None, validate_default=True
  )
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

  # The geo assignment and evaluation results are initially None, and are set
  # after the design is created, when the geos are assigned and the design is
  # evaluated.
  geo_assignment: GeoAssignment | None = None
  evaluation_results: ExperimentDesignEvaluationResults | None = None

  model_config = pydantic.ConfigDict(extra="forbid")

  def get_rng(self) -> np.random.Generator:
    """Returns a random number generator using the design's random seed."""
    return np.random.default_rng(self.random_seed)

  def get_summary_dict(
      self,
      target_power: float = 0.8,
      target_primary_metric_mde: float | None = None,
      use_relative_effects_where_possible: bool = True,
  ) -> dict[str, Any]:
    """Returns a summary dictionary of the experiment design."""
    if target_primary_metric_mde is not None:
      error_message = "Target MDE is not yet supported."
      logger.error(error_message)
      raise NotImplementedError(error_message)

    summary = {
        "design_id": self.design_id,
        "experiment_budget": str(self.experiment_budget),
        "primary_metric": self.primary_metric.name,
        "secondary_metrics": [m.name for m in self.secondary_metrics],
        "methodology": self.methodology,
        "runtime_weeks": self.runtime_weeks,
        "n_cells": self.n_cells,
        "cell_volume_constraint": str(self.cell_volume_constraint),
        "effect_scope": self.effect_scope.value,
        "alpha": self.alpha,
        "alternative_hypothesis": self.alternative_hypothesis,
        "random_seed": self.random_seed,
    }
    for group, geos in self.geo_eligibility.to_dict().items():
      if geos:
        summary[f"geo_eligibility_{group}"] = ", ".join(geos)

    if self.geo_assignment:
      for group, geos in self.geo_assignment.to_dict().items():
        summary[f"geo_assignment_{group}"] = ", ".join(geos)

    if self.evaluation_results:
      summary |= self.evaluation_results.get_summary_dict(
          target_power=target_power,
          use_relative_effects_where_possible=use_relative_effects_where_possible,
      )

    return summary

  def print_summary(
      self,
      target_power: float = 0.8,
      target_primary_metric_mde: float | None = None,
      use_relative_effects_where_possible: bool = True,
  ) -> None:
    """Prints a summary of the experiment design."""
    summary = self.get_summary_dict(
        target_power=target_power,
        target_primary_metric_mde=target_primary_metric_mde,
        use_relative_effects_where_possible=use_relative_effects_where_possible,
    )
    print(
        yaml.dump(
            summary,
            default_flow_style=False,
            sort_keys=False,
            indent=4,
        )
    )

  def make_variation(self, **kwargs: Any) -> "ExperimentDesign":
    """Creates a variation of the experiment design.

    This will create a new experiment design object with the same values as the
    original design, except for the values that are specified in the **kwargs.

    The design_id is set to a new UUID, and the evaluation_results and
    geo_assignment are reset to None.

    Args:
      **kwargs: The keyword arguments to update the design with.

    Returns:
      The new experiment design object.
    """
    kwargs["design_id"] = str(uuid.uuid4())
    kwargs["evaluation_results"] = None
    kwargs["geo_assignment"] = None

    model_copy = self.model_copy(deep=True).model_dump()
    variation = self.model_validate(model_copy | kwargs)

    return variation


def compare_designs(
    designs: list[ExperimentDesign],
    target_power: float = 0.8,
    target_primary_metric_mde: float | None = None,
    use_relative_effects_where_possible: bool = True,
) -> pd.DataFrame:
  """Create a dataframe with the summaries of each design for each comparison.

  Args:
    designs: The designs to compare.
    target_power: The target power for the experiment.
    target_primary_metric_mde: The target MDE for the primary metric.
    use_relative_effects_where_possible: Whether to use relative effects where
      possible.

  Returns:
    A dataframe with the summary of each design. The dataframe is indexed by the
    design id.
  """
  return pd.DataFrame([
      design.get_summary_dict(
          target_power=target_power,
          target_primary_metric_mde=target_primary_metric_mde,
          use_relative_effects_where_possible=use_relative_effects_where_possible,
      )
      for design in designs
  ]).set_index("design_id")
