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
"""API routes for working with saved designs."""

# pylint: disable=C0330, g-bad-import-order, g-multiple-import, g-importing-member
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Any
from services.design_storage import DesignStorageService

designs_router = APIRouter(prefix='/api/designs', tags=['designs'])


async def get_design_storage_service(request: Request) -> DesignStorageService:
  """Dependency to get the DesignStorageService from the app state."""
  return request.app.state.design_storage_service


@designs_router.get('', response_model=list[dict[str, Any]])
async def list_designs(storage_service: DesignStorageService = Depends(
    get_design_storage_service),):
  """Lists all available design files."""
  return await storage_service.list_designs()


@designs_router.get('/{design_id}', response_model=dict[str, Any])
async def get_design(
    design_id: str,
    storage_service: DesignStorageService = Depends(get_design_storage_service),
):
  """
  Retrieves a specific design file by its ID.
  """
  design_name = f"{design_id}.json"
  design = await storage_service.get_design(design_name)
  if design is None:
    raise HTTPException(status_code=404, detail='Design not found')
  return design


@designs_router.post('/{design_name}')
async def save_or_update_design(
    design_name: str,
    design_data: dict[str, Any],
    storage_service: DesignStorageService = Depends(get_design_storage_service),
):
  """Saves or updates a design file."""
  file_to_save = design_name
  if not file_to_save.endswith('.json'):
    file_to_save = f"{file_to_save}.json"

  result = await storage_service.save_design(file_to_save, design_data)
  if result is None:
    raise HTTPException(status_code=500, detail='Failed to save or update design')
  return result


@designs_router.delete('/{design_id}')
async def delete_design(
    design_id: str,
    storage_service: DesignStorageService = Depends(get_design_storage_service),
):
  """ Deletes a design file."""
  design_name = f"{design_id}.json"
  success = await storage_service.delete_design(design_name)
  if not success:
    raise HTTPException(status_code=500, detail='Failed to delete design')
  return {'status': 'success'}
