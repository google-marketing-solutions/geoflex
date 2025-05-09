"""A module for defining the exploration specification for geoflex."""

import logging
from typing import Annotated
from typing import Any
import geoflex.experiment_design
import pydantic

ExperimentBudget = geoflex.experiment_design.ExperimentBudget
EffectScope = geoflex.experiment_design.EffectScope

ValidatedMetric = geoflex.experiment_design.ValidatedMetric
ValidatedNCells = geoflex.experiment_design.ValidatedNCells
ValidatedExperimentBudget = geoflex.experiment_design.ValidatedExperimentBudget
ValidatedCellVolumeConstraint = (
    geoflex.experiment_design.ValidatedCellVolumeConstraint
)
ValidatedGeoEligibility = geoflex.experiment_design.ValidatedGeoEligibility
ValidatedAlpha = geoflex.experiment_design.ValidatedAlpha
ValidatedAlternativeHypothesis = (
    geoflex.experiment_design.ValidatedAlternativeHypothesis
)
ValidatedMetricList = geoflex.experiment_design.ValidatedMetricList
ensure_list = geoflex.experiment_design.ensure_list

logger = logging.getLogger(__name__)


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
def check_runtime_weeks_candidates_not_empty(
    runtime_weeks_candidates: list[int],
) -> list[int]:
  """Checks that the runtime weeks candidates are not empty."""
  if not runtime_weeks_candidates:
    error_message = "Runtime weeks candidates must not be empty."
    logger.error(error_message)
    raise ValueError(error_message)
  return runtime_weeks_candidates


ValidatedSeedList = Annotated[list[int], ensure_seed_is_list_and_not_empty]
ValidatedRuntimeWeeksCandidates = Annotated[
    list[int], ensure_list, check_runtime_weeks_candidates_not_empty
]
ValidatedEligibleMethodologies = Annotated[list[str], ensure_list]
ValidatedGeoEligibilityCandidates = Annotated[
    list[ValidatedGeoEligibility], ensure_list
]
ValidatedCellVolumeConstraintCandidates = Annotated[
    list[ValidatedCellVolumeConstraint], ensure_list
]
ValidatedExperimentBudgetCandidates = Annotated[
    list[ValidatedExperimentBudget],
    ensure_list,
    drop_extra_budget_candidates_if_no_cost_metrics,
]


class ExperimentDesignExplorationSpec(pydantic.BaseModel):
  """All the inputs needed for geoflex to design an experiment.

  This includes some parameters of the experiment, such as the experiment type
  and the metrics. It also includes constraints on the design, such as the
  maximum and minimum number of weeks, the number of cells, and the number of
  geos per group.

  Attributes:
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
    methodology_parameter_candidates: A dictionary of methodology parameter
      candidates. The key is the methodology name, and the value is a dictionary
      of parameter name and parameter candidates. The parameter candidates are a
      list of values that can be used for that parameter and methodology. If a
      parameter is not specified then the methodology will suggest a value for
      it from the default candidates.
  """

  primary_metric: ValidatedMetric

  runtime_weeks_candidates: ValidatedRuntimeWeeksCandidates = [4]
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
  methodology_parameter_candidates: dict[str, dict[str, list[Any]]] = {}

  model_config = pydantic.ConfigDict(extra="forbid")
