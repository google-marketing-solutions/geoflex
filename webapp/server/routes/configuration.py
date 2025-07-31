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
"""API routes for working with app configuration."""

# pylint: disable=C0330, g-bad-import-order, g-multiple-import, g-importing-member
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from config import get_config, save_config
from googleapiclient.errors import HttpError

router = APIRouter(prefix='/api/config', tags=['configuration'])


class ConfigModel(BaseModel):
  spreadsheet_id: str


class ShareResponse(BaseModel):
  success: bool
  message: str


# Get configuration handler
@router.get('')
async def get_configuration():
  """Get current application configuration."""
  config = get_config(fail_if_not_exists=False)
  return config.to_dict()


@router.post('/recreate', response_model=ConfigModel)
async def recreate_spreadsheet(request: Request):
  """Recreate the master spreadsheet."""
  # pylint: disable=C0415, g-import-not-at-top
  from services.datasources import DataSourceService

  datasource_service: DataSourceService = request.app.state.datasource_service

  try:
    datasource_service.config.spreadsheet_id = ''
    spreadsheet_id = await datasource_service.initialize_master_spreadsheet()
    if hasattr(request.state, 'iap_user') and request.state.iap_user:
      await datasource_service.share_master_spreadsheet(request.state.iap_user)
    return ConfigModel(spreadsheet_id=spreadsheet_id)
  except HttpError as error:
    raise HTTPException(
        status_code=500,
        detail=f'Error creating spreadsheet: {str(error)}') from error


# Update configuration handler
@router.put('', response_model=ConfigModel)
async def update_configuration(config_data: ConfigModel, request: Request):
  """Update application configuration."""
  # Get current config
  current_config = get_config()

  current_config.spreadsheet_id = config_data.spreadsheet_id

  # Save updated config
  save_config(current_config)

  # Update the datasource service reference
  if hasattr(request.app.state, 'datasource_service'):
    request.app.state.datasource_service.config.spreadsheet_id = (
        config_data.spreadsheet_id)

  return ConfigModel(spreadsheet_id=config_data.spreadsheet_id)


# Share spreadsheet with current user
@router.post('/share', response_model=ShareResponse)
async def share_spreadsheet(request: Request):
  """Share the master spreadsheet with the current user."""
  # pylint: disable=C0415, g-import-not-at-top
  from services.datasources import DataSourceService
  # Get the current user from middleware
  if not hasattr(request.state, 'iap_user') or not request.state.iap_user:
    raise HTTPException(
        status_code=400,
        detail='User not authenticated or user email not available')

  user_email = request.state.iap_user

  datasource_service: DataSourceService = request.app.state.datasource_service

  try:
    await datasource_service.share_master_spreadsheet(user_email)

    return ShareResponse(
        success=True, message=f'Spreadsheet shared with {user_email}')
  except HttpError as error:
    raise HTTPException(
        status_code=500,
        detail=f'Error sharing spreadsheet: {str(error)}') from error
