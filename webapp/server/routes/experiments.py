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
import os
import sys
from typing import List, Optional, Any
import pydantic
from fastapi import APIRouter, HTTPException, Depends, Request

vendor_dir = os.path.join(os.path.dirname(__file__), '../lib')
if vendor_dir not in sys.path:
  sys.path.insert(0, vendor_dir)

from geoflex.experiment_design import ExperimentDesign, ExperimentType, GeoAssignment, ExperimentDesignConstraints
from geoflex.metrics import Metric
from geoflex.methodology import MethodologyName

from services.datasources import DataSourceService
from logger import logger

router = APIRouter(prefix='/api/experiments', tags=['experiments'])


class ExplorationRequest(pydantic.BaseModel):
  """Request model for experiment design exploration."""

  # Datasource identifier
  datasource_id: str

  # Core parameters
  experiment_type: ExperimentType
  primary_metric: str

  # Test parameters
  n_cells: int = 2
  alpha: float = 0.1
  alternative_hypothesis: str = 'two-sided'

  # Duration parameters
  min_runtime_weeks: int
  max_runtime_weeks: int

  # Methodology options (empty means explore all)
  methodologies: List[str] = []

  # Geo constraints
  fixed_geos: Optional[dict[str, list[str] | list[list[str]]]] = None

  # Optimization target
  optimization_target: str = 'power'  # "power" or "mde"
  target_power: float | None = None  # Used when optimization_target is "power"
  target_mde: float | None = None  # Used when optimization_target is "mde"

  # Optional advanced parameters
  pretest_weeks: Optional[int] = None
  trimming_quantile_candidates: List[float] = [0.0]

  model_config = pydantic.ConfigDict(extra='forbid')


class DesignSummary(pydantic.BaseModel):
  """Summary of an experiment design for the UI."""

  design_id: str
  methodology: str
  power: float  # As a percentage (0-100)
  mde: float  # As a percentage
  duration: int  # Runtime in weeks
  parameters: dict[str, Any]  # All parameters used for this design
  groups: dict[
      str, list[str]]  # dict keyed by 'Control', 'Test' with lists of geo units

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

  # Get datasource
  datasource = await ds_service.get_datasource_by_id(request.datasource_id)
  datasource_data = await ds_service.load_datasource_data(datasource.id)
  geo_column = datasource.columns.geo_column
  all_geos = list(set(row[geo_column] for row in datasource_data))

  # Handle fixed geo assignments if provided
  excluded_geos = []
  fixed_control_geos = []
  fixed_test_geos = []

  if request.fixed_geos:
    excluded_geos = request.fixed_geos.get('exclude', [])
    fixed_control_geos = request.fixed_geos.get('control', [])

    # Handle treatment assignments - might be nested lists for multi-cell
    treatment = request.fixed_geos.get('treatment', [])
    if treatment and isinstance(treatment[0], list):
      # Multi-cell case
      for treatment_group in treatment:
        fixed_test_geos.extend(treatment_group)
    elif treatment:
      # Single-cell case
      fixed_test_geos.extend(treatment)

  # Remove excluded and fixed geos from the available pool
  available_geos = [
      geo for geo in all_geos if geo not in excluded_geos and
      geo not in fixed_control_geos and geo not in fixed_test_geos
  ]

  # constraints = ExperimentDesignConstraints(
  #     experiment_type=request.experiment_type,
  #     max_runtime_weeks=request.max_runtime_weeks,
  #     min_runtime_weeks=request.min_runtime_weeks,
  #     n_cells=request.n_cells,
  #     fixed_geos=fixed_geos,
  #     trimming_quantile_candidates=request.trimming_quantile_candidates,
  #     # TODO: n_geos_per_group_candidates
  # )

  # Determine methodologies to explore
  methodologies_to_explore = request.methodologies or [
      MethodologyName.TBR_MM, MethodologyName.TBR, MethodologyName.TM,
      MethodologyName.GBR
  ]

  # Create primary and secondary metrics
  primary_metric = Metric(name=request.primary_metric)
  # secondary_metrics = [
  #     Metric(name=metric) for metric in request.secondary_metrics
  # ]

  # TODO: For now, return mock data
  designs = []
  for i, methodology in enumerate(methodologies_to_explore):
    # Create basic parameters
    power = 90.0 - (i * 5.0)  # Mock decreasing power
    mde = (request.target_mde or 0.1) + (i * 0.5)  # Mock increasing MDE
    duration = min(request.min_runtime_weeks + (i * 2),
                   request.max_runtime_weeks)

    # Create different splits based on methodology
    available_count = len(available_geos)
    control_count = max(3, available_count // 2)

    if methodology == 'TBR-MM':
      control_pool = available_geos[:control_count]
      test_pool = available_geos[control_count:]
    elif methodology == 'TBR':
      control_pool = available_geos[::2][:control_count]
      test_pool = [g for g in available_geos if g not in control_pool]
    elif methodology == 'TM':
      control_pool = available_geos[1::3][:control_count]
      test_pool = [g for g in available_geos if g not in control_pool]
    else:
      control_pool = available_geos[2::3][:control_count]
      test_pool = [g for g in available_geos if g not in control_pool]

    # Add fixed geo assignments
    control_geos = fixed_control_geos + control_pool
    test_geos = fixed_test_geos + test_pool

    # Create the design summary
    design = DesignSummary(
        design_id=f'design_{i}',
        methodology=methodology,
        power=power,
        mde=mde,
        duration=duration,
        parameters={
            'methodology': methodology,
            'runtime_weeks': duration,
            'experiment_type': request.experiment_type,
            'n_cells': request.n_cells,
            'alpha': request.alpha,
            'alternative_hypothesis': request.alternative_hypothesis,
            'primary_metric': request.primary_metric
        },
        groups={
            'Control': [geo for geo in control_geos],
            'Test': [geo for geo in test_geos]
        })

    designs.append(design)

  return ExplorationResponse(designs=designs)
