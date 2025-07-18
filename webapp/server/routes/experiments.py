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
import math
from typing import List, Literal, Optional, Union
import geoflex
import pandas as pd
import pydantic
from fastapi import APIRouter, Depends, HTTPException, Request
from geoflex.methodology import list_methodologies
from geoflex.metrics import CPiA, Metric, iROAS
from logger import logger
from services.datasources import DataSourceService

router = APIRouter(prefix='/api/experiments', tags=['experiments'])


class Experiment:
  pass


class ExperimentBudget(pydantic.BaseModel):
  value: float
  budget_type: geoflex.ExperimentBudgetType


# --- Metric Models ---


class CustomMetric(pydantic.BaseModel):
  type: Literal['custom'] = 'custom'
  name: Optional[str] = None
  column: Optional[str] = None
  cost_column: Optional[str] = None
  metric_per_cost: bool = False
  cost_per_metric: bool = False


class IROASMetric(pydantic.BaseModel):
  type: Literal['iroas'] = 'iroas'
  #name: str = 'iROAS'
  return_column: Optional[str] = None
  cost_column: Optional[str] = None


class CPIAMetric(pydantic.BaseModel):
  type: Literal['cpia'] = 'cpia'
  #name: str = 'CPiA'
  conversions_column: Optional[str] = None
  cost_column: Optional[str] = None


AnyMetric = Union[str, CustomMetric, IROASMetric, CPIAMetric]


class FixedGeos(pydantic.BaseModel):
  control: List[str] = []
  treatment: List[List[str]] = []
  exclude: List[str] = []


class ExplorationRequest(pydantic.BaseModel):
  """Request model for experiment design exploration."""

  # Datasource identifier
  datasource_id: str

  # Core parameters
  primary_metric: AnyMetric
  secondary_metrics: List[AnyMetric] = []

  # Test parameters
  n_cells: int = 2
  alpha: float = 0.1
  alternative_hypothesis: Literal['two-sided', 'one-sided'] = 'two-sided'

  budgets: list[ExperimentBudget]

  # Duration parameters
  min_runtime_weeks: int
  max_runtime_weeks: int

  # Methodology options (empty means explore all)
  methodologies: List[str] = []

  # Geo constraints
  fixed_geos: Optional[FixedGeos] = None

  target_power: float | None = None

  cell_volume_constraint: Optional[geoflex.CellVolumeConstraint] = None

  # Optional advanced parameters
  trimming_quantile_candidates: List[float] = [0.0]

  effect_scope: geoflex.EffectScope
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


def _parse_metric(metric_request: AnyMetric) -> Metric:
  """Parses a metric request model into a geoflex.metrics.Metric object."""
  if isinstance(metric_request, str):
    return Metric(name=metric_request)
  elif isinstance(metric_request, CustomMetric):
    metric_args = {
        'name': metric_request.name,
        'column': metric_request.column,
        'metric_per_cost': metric_request.metric_per_cost,
        'cost_per_metric': metric_request.cost_per_metric,
    }
    if not metric_request.name:
      raise ValueError('Metric has no name specified')
    if metric_request.cost_column:
      metric_args['cost_column'] = metric_request.cost_column
    return Metric(**metric_args)
  elif isinstance(metric_request, IROASMetric):
    if not metric_request.return_column:
      raise ValueError('Metric iROAS has no return_column specified')
    if not metric_request.cost_column:
      raise ValueError('Metric iROAS has no cost_column specified')
    return iROAS(
        return_column=metric_request.return_column,
        cost_column=metric_request.cost_column,
    )
  elif isinstance(metric_request, CPIAMetric):
    if not metric_request.conversions_column:
      raise ValueError('Metric CPiA has no conversions_column specified')
    if not metric_request.cost_column:
      raise ValueError('Metric CPiA has no cost_column specified')
    return CPiA(
        conversions_column=metric_request.conversions_column,
        cost_column=metric_request.cost_column,
    )
  # This will not be called as Pydantic validates the input
  raise ValueError('Unknown metric type')


@router.post('/explore', response_model=ExplorationResponse)
async def explore_experiment_designs(
    request: ExplorationRequest,
    ds_service: DataSourceService = Depends(get_datasource_service),
) -> ExplorationResponse:
  """Explores experiment designs based on the given constraints.

    This endpoint takes the datasource and experiment parameters and returns
    a list of possible experiment designs, ranked by their statistical power.
    """
  logger.debug('Generating experiment designs')
  logger.debug(request.model_dump())
  # Get datasource
  datasource = await ds_service.get_datasource_by_id(request.datasource_id)
  if not datasource or not datasource.id:
    raise HTTPException(
        status_code=404,
        detail=f'Datasource with id {request.datasource_id} not found or is invalid',
    )
  datasource_data = await ds_service.load_datasource_data(datasource.id)

  datasource_pd = pd.DataFrame(datasource_data)
  historical_data = geoflex.GeoPerformanceDataset(
      data=datasource_pd,
      geo_id_column=datasource.columns.geo_column,
      date_column=datasource.columns.date_column,
  )

  # Handle fixed geo assignments if provided
  excluded_geos = set()
  fixed_control_geos = set()
  fixed_test_geos = []

  if request.fixed_geos:
    excluded_geos = set(request.fixed_geos.exclude)
    fixed_control_geos = set(request.fixed_geos.control)
    fixed_test_geos = [set(geos) for geos in request.fixed_geos.treatment]

  # TODO: remove TestingMethodology
  methodologies_to_explore = (
      request.methodologies or list_methodologies() or ['TestingMethodology'])

  primary_metric = _parse_metric(request.primary_metric)
  secondary_metrics = [
      _parse_metric(metric) for metric in request.secondary_metrics
  ]
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
      secondary_metrics=secondary_metrics,
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
              exclude=excluded_geos,
          ),
      ],
      effect_scope=request.effect_scope,
      # TODO: failing with []
      # cell_volume_constraint_candidates=[request.cell_volume_constraint]
      # if request.cell_volume_constraint else [None],
  )
  logger.debug(exploration_spec.model_dump())
  design_explorer = geoflex.ExperimentDesignExplorer(
      explore_spec=exploration_spec,
      historical_data=historical_data,
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
    mde = None
    if design.evaluation_results:
      mdes = design.evaluation_results.get_mde(
          target_power=target_power, relative=True, aggregate_across_cells=True)
      if mdes and primary_metric.name in mdes:
        mde_value = mdes[primary_metric.name]
        if isinstance(mde_value, list):
          mde = [
              v * 100 if v is not None and math.isfinite(v) else None
              for v in mde_value
          ]
        elif mde_value is not None and math.isfinite(mde_value):
          mde = mde_value * 100

    logger.debug(design.get_summary_dict())
    # wrap library's ExperimentDesign into our class with mde
    design_summary = DesignSummary(**design.model_dump(), mde=mde)
    designs.append(design_summary)

  return ExplorationResponse(designs=designs)
