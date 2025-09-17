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
"""API routes for working with data sources."""

# pylint: disable=C0330, g-bad-import-order, g-multiple-import, g-importing-member
from fastapi import APIRouter, HTTPException, Depends, Request, Response
from pydantic import BaseModel
from typing import Any
import csv
import io
from models.datasources import DataSource, ColumnSchema
from services.datasources import DataSourceService
from logger import logger

router = APIRouter(prefix='/api/datasources', tags=['datasources'])


# Dependency to get the datasource service from app state
async def get_datasource_service(request: Request) -> DataSourceService:
  """Get the initialized DataSourceService from app state."""
  return request.app.state.datasource_service


@router.get('', response_model=list[DataSource])
async def get_datasources(service: DataSourceService = Depends(
    get_datasource_service)) -> list[DataSource]:
  """Return a list of all data sources."""
  return await service.get_datasources()


class DataSourceUpdate(BaseModel):
  """Request model for update_datasource and create_datasource methods."""
  id: str | None = None
  name: str
  description: str | None = None
  source_link: str
  columns: ColumnSchema
  data: list[dict[str, Any]]


@router.post('', response_model=DataSource)
async def create_datasource(
    req: DataSourceUpdate,
    service: DataSourceService = Depends(get_datasource_service)
) -> DataSource:
  """Save a new data source."""
  ds = DataSource(
      id=req.id,
      name=req.name,
      description=req.description,
      columns=req.columns,
      source_link=req.source_link,
      created_at=None,
      updated_at=None,
      data=req.data)
  return await service.add_datasource(ds)


@router.put('/{id}', response_model=DataSource)
# pylint: disable=W0622
async def update_datasource(
    id: str,
    req: DataSourceUpdate,
    service: DataSourceService = Depends(get_datasource_service)
) -> DataSource:
  """Update a data source."""
  ds = DataSource(
      id=id,
      name=req.name,
      description=req.description,
      columns=req.columns,
      source_link=req.source_link,
      data=req.data,
      created_at=None,
      updated_at=None)
  updated = await service.update_datasource(ds)
  if not updated:
    raise HTTPException(status_code=404, detail='Data source not found')
  return updated


@router.delete('/{id}')
# pylint: disable=W0622
async def delete_datasource(
    id: str, service: DataSourceService = Depends(get_datasource_service)):
  """Delete an existing data source by its id."""
  deleted = await service.delete_datasource(id)
  if not deleted:
    raise HTTPException(status_code=404, detail='Data source not found')
  return {'status': 'success'}


@router.get('/{id}/data')
# pylint: disable=W0622
async def get_datasource_data(
    id: str, service: DataSourceService = Depends(get_datasource_service)
) -> list[dict[str, Any]]:
  logger.debug('loading data for data source %s', id)
  data = await service.load_datasource_data(id)
  if not data:
    raise HTTPException(status_code=404, detail='Data not found')
  return data


@router.get('/{id}/download')
# pylint: disable=W0622
async def download_datasource_data(
    id: str, service: DataSourceService = Depends(get_datasource_service)
) -> Response:
  """Download the data source's data as a CSV file."""
  logger.debug('downloading data for data source %s', id)
  data = await service.load_datasource_data(id)
  if not data:
    raise HTTPException(status_code=404, detail='Data not found')

  # Create a string buffer to hold the CSV data
  output = io.StringIO()
  writer = csv.writer(output)

  # Write the header row
  if data:
    writer.writerow(data[0].keys())
    # Write the data rows
    for row in data:
      writer.writerow(row.values())

  # Get the CSV data as a string
  csv_data = output.getvalue()

  # Return the CSV data as a response
  filename = f'{id}.csv'
  return Response(
      content=csv_data,
      media_type='text/csv',
      headers={
          'Content-Disposition': f'attachment; filename={filename}',
          'filename': filename
      })


@router.get('/preview')
async def preview_datasource_data(
    url: str, service: DataSourceService = Depends(get_datasource_service)):
  """Preview data from an external Google Sheets URL."""
  try:
    data = await service.preview_external_data(url)
    return data
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e)) from e
  except Exception as e:
    logger.error('Error previewing external data: %s', e)
    raise HTTPException(
        status_code=500, detail='Failed to preview external data') from e
