# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""API routes for working with experiments."""

# pylint: disable=C0330, g-bad-import-order, g-multiple-import, g-importing-member
from typing import List, Optional, Literal
import pydantic
import math
from fastapi import APIRouter, Depends, Request
import pandas as pd
import geoflex

from geoflex.metrics import Metric
from geoflex.methodology import list_methodologies

from services.datasources import DataSourceService
from logger import logger

router = APIRouter(prefix='/api/experiments', tags=['experiments'])


class Experiment:
  pass


class ExperimentBudget(pydantic.BaseModel):
  value: float
  budget_type: geoflex.ExperimentBudgetType


class ExplorationRequest(pydantic.BaseModel):
  """Request model for experiment design exploration."""

  # Datasource identifier
  datasource_id: str

  # Core parameters
  primary_metric: str

  # Test parameters
  n_cells: int = 2
  alpha: float = 0.1
  alternative_hypothesis: Literal['two-sides', 'one-sided'] = 'two-sided'

  budgets: list[ExperimentBudget]

  # Duration parameters
  min_runtime_weeks: int
  max_runtime_weeks: int

  # Methodology options (empty means explore all)
  methodologies: List[str] = []

  # Geo constraints
  fixed_geos: Optional[dict[str, list[str] | list[list[str]]]] = None

  target_power: float | None = None

  # Optional advanced parameters
  trimming_quantile_candidates: List[float] = [0.0]

  effect_scope: geoflex.EffectScope
  simulations_per_trial: int | None = None
  max_trials: int | None = None
  n_designs: int | None = None

  model_config = pydantic.ConfigDict(extra='forbid')


class DesignSummary(geoflex.ExperimentDesign):
  """Summary of an experiment design for the UI."""
  mde: list[float | None] | float | None

  model_config = pydantic.ConfigDict(extra='forbid')


class ExplorationResponse(pydantic.BaseModel):
  """Response model for experiment design exploration."""

  designs: List[DesignSummary]

  model_config = pydantic.ConfigDict(extra='forbid')


# Dependency to get the datasource service from app state
async def get_datasource_service(request: Request) -> DataSourceService:
  """Get the initialized DataSourceService from app state."""
  return request.app.state.datasource_service


@router.post('/explore', response_model=ExplorationResponse)
async def explore_experiment_designs(
    request: ExplorationRequest,
    ds_service: DataSourceService = Depends(get_datasource_service)
) -> ExplorationResponse:
  """Explores experiment designs based on the given constraints.

  This endpoint takes the datasource and experiment parameters and returns
  a list of possible experiment designs, ranked by their statistical power.
  """
  logger.debug('Generating test designs')
  logger.debug(request.model_dump())
  # Get datasource
  datasource = await ds_service.get_datasource_by_id(request.datasource_id)
  datasource_data = await ds_service.load_datasource_data(datasource.id)

  datasource_pd = pd.DataFrame(datasource_data)
  historical_data = geoflex.GeoPerformanceDataset(
      data=datasource_pd,
      geo_id_column=datasource.columns.geo_column,
      date_column=datasource.columns.date_column)

  # Handle fixed geo assignments if provided
  excluded_geos = []
  fixed_control_geos = []
  fixed_test_geos = []

  if request.fixed_geos:
    excluded_geos = request.fixed_geos.get('exclude', [])
    fixed_control_geos = request.fixed_geos.get('control', [])
    fixed_test_geos = request.fixed_geos.get('treatment', [])

  # TODO: remove TestingMethodology
  methodologies_to_explore = request.methodologies or list_methodologies() or [
      'TestingMethodology'
  ]

  primary_metric = Metric(name=request.primary_metric)
  # secondary_metrics = [
  #     Metric(name=metric) for metric in request.secondary_metrics
  # ]
  experiment_budget_candidates = []
  if request.budgets:
    for budget in request.budgets:
      experiment_budget_candidates.append(
          geoflex.ExperimentBudget(
              value=budget.value, budget_type=budget.budget_type))
  exploration_spec = geoflex.ExperimentDesignExplorationSpec(
      primary_metric=primary_metric,
      alternative_hypothesis='greater' if request.alternative_hypothesis
      == 'one-sided' else request.alternative_hypothesis,
      # secondary_metrics=[
      #     "conversions",
      #     "revenue",
      #     geoflex.metrics.CPA(
      #         conversions_column="conversions",
      #         cost_column="cost"
      #     ),
      # ],
      alpha=request.alpha,
      experiment_budget_candidates=experiment_budget_candidates,
      eligible_methodologies=methodologies_to_explore,
      runtime_weeks_candidates=[request.min_runtime_weeks]
      if request.min_runtime_weeks == request.max_runtime_weeks else list(
          range(request.min_runtime_weeks, request.max_runtime_weeks + 1)),
      n_cells=request.n_cells,
      geo_eligibility_candidates=[
          geoflex.experiment_design.GeoEligibility(
              control=fixed_control_geos,
              treatment=fixed_test_geos,
              exclude=excluded_geos),
      ],
      effect_scope=request.effect_scope
      # TODO: cell_volume_constraint_candidates=[
      #     None,
      #     geoflex.CellVolumeConstraint(
      #         values=[5, 5, 5],
      #         constraint_type=geoflex.CellVolumeConstraintType.MAX_GEOS
      #     )
      # ],
  )
  logger.debug(exploration_spec.model_dump())
  design_explorer = geoflex.ExperimentDesignExplorer(
      explore_spec=exploration_spec,
      historical_data=historical_data,
      simulations_per_trial=request.simulations_per_trial
      if request.simulations_per_trial > 1 else 100,
  )
  design_explorer.explore(max_trials=request.max_trials or 100)

  top_designs = design_explorer.get_designs(top_n=request.n_designs or 5)
  designs = []
  target_power = request.target_power or 0.8
  if target_power > 1:
    target_power = target_power / 100
  # library's response doesn't have MDE needed on the client,
  # we have to fetch and return them separately
  for design in top_designs:
    mdes = design.evaluation_results.get_mde(
        target_power=target_power, relative=True, aggregate_across_cells=True)
    logger.debug(design.get_summary_dict())
    # wrap library's ExperimentDesign into our class with mde
    mde = mdes[primary_metric.name]
    if isinstance(mde, list):
      mde = [v if math.isfinite(v) else None for v in mde]
    design = DesignSummary(
        **design.model_dump(), mde=mde if math.isfinite(mde) else None)
    designs.append(design)

  return ExplorationResponse(designs=designs)
