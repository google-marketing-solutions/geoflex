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
"""Data source service."""

# pylint: disable=C0330, g-bad-import-order, g-multiple-import, g-importing-member
from typing import Any
from datetime import datetime, date
import uuid
import re
import google.auth
from google.auth import credentials
from googleapiclient.discovery import build
from models.datasources import DataSource, ColumnSchema
from logger import logger
from config import get_config, save_config


def get_credentials(scopes) -> credentials.Credentials:
  cred, _ = google.auth.default(scopes)
  return cred


DATASOURCE_COLUMNS = [
    'id', 'name', 'description', 'created_at', 'updated_at', 'source_link',
    'columns'
]
idx_id = DATASOURCE_COLUMNS.index('id')  # A
idx_name = DATASOURCE_COLUMNS.index('name')  # B
idx_description = DATASOURCE_COLUMNS.index('description')  # C
idx_created = DATASOURCE_COLUMNS.index('created_at')  # D
idx_updated = DATASOURCE_COLUMNS.index('updated_at')  # E
idx_source_link = DATASOURCE_COLUMNS.index('source_link')  # F
idx_columns = DATASOURCE_COLUMNS.index('columns')  # G

iso_pattern = re.compile(r'^(\d{4})-(\d{1,2})-(\d{1,2})$')


class DataSourceService:
  """Service for working with Data Sources (definitions and data)."""

  def __init__(self):
    # Initialize Google Sheets API
    self._init_sheets_api()

    self.config = get_config(fail_if_not_exists=False)
    self.datasources_sheet_name = 'DataSources'

  def _init_sheets_api(self):
    """Initialize Google Sheets API client."""
    scopes = ['https://www.googleapis.com/auth/spreadsheets']
    self.sheets_service = build(
        'sheets', 'v4', credentials=get_credentials(scopes))

  def _parse_spreadsheet_row(self, row: list[Any]) -> DataSource:
    """Parse a row values from Sheets API into DataSource."""
    column_parts = row[idx_columns].split(':')

    # Create the ColumnSchema
    column_schema = ColumnSchema(
        geo_column=column_parts[0] if len(column_parts) > 0 else '',
        date_column=column_parts[1] if len(column_parts) > 1 else '',
        metric_columns=column_parts[2].split(',')
        if len(column_parts) > 2 else [],
        cost_column=column_parts[3] if len(column_parts) > 3 else None)

    # Create the DataSource with nested column structure
    datasource = DataSource(
        id=row[idx_id],
        name=row[idx_name],
        description=row[idx_description],
        created_at=datetime.fromisoformat(row[idx_created]),
        updated_at=datetime.fromisoformat(row[idx_updated]),
        source_link=row[idx_source_link],
        columns=column_schema)
    return datasource

  async def get_datasources(self) -> list[DataSource]:
    """Get all data sources from master spreadsheet.

    Returns:
      List of data sources with data.
    """
    logger.debug('Loading all data sources')
    result = self.sheets_service.spreadsheets().values().get(
        spreadsheetId=self.config.spreadsheet_id,
        range=f'{self.datasources_sheet_name}!A2:G').execute()

    values = result.get('values', [])
    datasources = []

    for row in values:
      if len(row) >= len(
          DATASOURCE_COLUMNS):  # Ensure we have all needed columns
        # Parse the columns string
        datasource = self._parse_spreadsheet_row(row)
        datasources.append(datasource)

    return datasources

  # pylint: disable=W0622
  async def get_datasource_by_id(self,
                                 id: str,
                                 throw_if_not_found=False) -> DataSource | None:
    """Get a specific data source by ID.

    Args:
      id: data source id.
      throw_if_not_found: true to raise ValueError if data source not found.

    Returns:
      Data source (without data).

    Raises:
      ValueError: If datasource not found and throw_if_not_found=true.
    """
    # Otherwise fetch all and update cache
    datasources = await self.get_datasources()
    for ds in datasources:
      if ds.id == id:
        return ds

    if throw_if_not_found:
      raise ValueError(f'Data source with id={id} not found')
    return None

  def _get_storage_column_config(self, columns: ColumnSchema) -> str:
    config = f'{columns.geo_column}:{columns.date_column}:' + ','.join(
        columns.metric_columns)
    if columns.cost_column:
      config += f':{columns.cost_column}'
    return config

  async def add_datasource(self, datasource: DataSource) -> DataSource:
    """Add a new data source to the master spreadsheet.

    Args:
      datasource: data source to update.

    Returns:
      Updated data source (without data).
    """
    # Generate new ID
    new_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    column_config = self._get_storage_column_config(datasource.columns)
    datasource.id = new_id
    datasource.created_at = datetime.fromisoformat(now)
    datasource.updated_at = datetime.fromisoformat(now)

    # Prepare row data
    row_data = [None] * len(DATASOURCE_COLUMNS)
    row_data[idx_id] = datasource.id
    row_data[idx_name] = datasource.name
    row_data[idx_description] = datasource.description
    row_data[idx_created] = now
    row_data[idx_updated] = now
    row_data[idx_source_link] = datasource.source_link
    row_data[idx_columns] = column_config
    logger.debug(row_data)
    # Write to Google Sheets
    self.sheets_service.spreadsheets().values().append(
        spreadsheetId=self.config.spreadsheet_id,
        range=f'{self.datasources_sheet_name}!A2:G',
        valueInputOption='RAW',
        insertDataOption='INSERT_ROWS',
        body={
            'values': [row_data]
        }).execute()

    # If data is provided, save the data
    if datasource.data:
      await self._save_datasource_data(datasource)

    return datasource

  async def update_datasource(self, datasource: DataSource) -> DataSource:
    """Update an existing data source.

    Args:
      datasource: data source to update (data will be ignored).

    Returns:
      Updated data source.

    Raises:
      ValueError: If datasource not found.
    """
    # Find row to update
    result = self.sheets_service.spreadsheets().values().get(
        spreadsheetId=self.config.spreadsheet_id,
        range=f'{self.datasources_sheet_name}!A2:G').execute()
    row_idx = None
    rows = result.get('values', [])
    for i, row in enumerate(rows):
      if row and row[0] == datasource.id:
        row_idx = i
        break

    if row_idx is None:
      raise ValueError(f'Data source with id={datasource.id} not found')

    existing = self._parse_spreadsheet_row(rows[row_idx])

    now = datetime.now().isoformat()
    column_config = self._get_storage_column_config(datasource.columns)

    # Update row in the spreadsheet
    values = [None] * len(DATASOURCE_COLUMNS)
    values[idx_id] = datasource.id
    values[idx_name] = datasource.name
    values[idx_description] = datasource.description
    values[idx_created] = existing.created_at.isoformat()
    values[idx_updated] = now
    values[idx_source_link] = datasource.source_link
    values[idx_columns] = column_config
    row_idx = row_idx + 2  # Sheets API is 1-indexed + 1st row for headers
    self.sheets_service.spreadsheets().values().update(
        spreadsheetId=self.config.spreadsheet_id,
        range=f'{self.datasources_sheet_name}!A{row_idx}:G{row_idx}',
        valueInputOption='RAW',
        body={
            'values': [values]
        }).execute()

    # Create updated datasource object
    updated_datasource = DataSource(
        id=datasource.id,
        name=datasource.name,
        description=datasource.description,
        created_at=existing.created_at,
        updated_at=datetime.fromisoformat(now),
        source_link=datasource.source_link,
        columns=datasource.columns)

    return updated_datasource

  async def _save_datasource_data(self, datasource: DataSource) -> bool:
    """Internal method to save data for a data source."""
    if not datasource.data:
      raise ValueError(
          f'Data source {datasource.name} ({datasource.id}) has no data')

    if not isinstance(datasource.data, list):
      raise ValueError(f'Invalid data format for datasource {datasource.id}')

    # Determine the sheet name for this datasource
    sheet_name = f'data_{datasource.id}'

    try:
      # Check if sheet exists already
      sheet_exists = False
      sheet_metadata = self.sheets_service.spreadsheets().get(
          spreadsheetId=self.config.spreadsheet_id).execute()

      for sheet in sheet_metadata.get('sheets', []):
        if sheet.get('properties', {}).get('title') == sheet_name:
          sheet_exists = True
          break

      # Create sheet if it doesn't exist
      if not sheet_exists:
        self.sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=self.config.spreadsheet_id,
            body={
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': sheet_name
                        }
                    }
                }]
            }).execute()
      else:
        # Clear existing data if sheet exists
        self.sheets_service.spreadsheets().values().clear(
            spreadsheetId=self.config.spreadsheet_id,
            range=f'{sheet_name}!A:Z').execute()

      # We'll save all data as is, regardless of column schema,
      # because later the use can change their column configuration
      data_rows = datasource.data

      if not data_rows:
        return

      all_keys = []
      for d in data_rows:
        for key in d:
          if key not in all_keys:
            if key != '__index':
              all_keys.append(key)

      # Convert to list of lists with headers as first row
      values = [all_keys]  # First row is headers

      # Add data rows
      for row in data_rows:

        if val := row[datasource.columns.date_column]:
          if isinstance(val, datetime):
            row[datasource.columns.date_column] = val.date().isoformat()
          elif isinstance(val, str):
            if match := re.match(iso_pattern, val):
              # format is correct, normalize it just in case
              year, month, day = match.groups()
              row[datasource.columns.date_column] = (
                  f'{int(year):04d}-{int(month):02d}-{int(day):02d}')
            else:
              raise ValueError(
                  f'Value {val} for date column ({datasource.columns.date_column}) is not in valid format (yyyy-mm-dd)'
              )
        row_values = [row.get(key, None) for key in all_keys]
        values.append(row_values)

      # Write to Google Sheets
      self.sheets_service.spreadsheets().values().update(
          spreadsheetId=self.config.spreadsheet_id,
          range=f'{sheet_name}!A1',
          valueInputOption='RAW',
          body={
              'values': values
          }).execute()

    except Exception as e:
      raise ValueError(f'Error saving datasource data: {str(e)}') from e

  # pylint: disable=W0622
  async def delete_datasource(self, id: str) -> bool:
    """Delete a data source.

    Args:
      id: data source id.

    Returns:
      true if data source was deleted.
    """
    # First get the datasource
    datasource = await self.get_datasource_by_id(id)
    if not datasource:
      return False

    # Find row to delete
    result = self.sheets_service.spreadsheets().values().get(
        spreadsheetId=self.config.spreadsheet_id,
        range=f'{self.datasources_sheet_name}!A:A').execute()

    row_idx = None
    ids = result.get('values', [])
    for i, row_id in enumerate(ids):
      if row_id and row_id[0] == id:
        row_idx = i + 1  # Sheets API is 1-indexed
        break

    if row_idx is None:
      return False

    # Delete row (using batch update to delete rows)
    self.sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=self.config.spreadsheet_id,
        body={
            'requests': [{
                'deleteDimension': {
                    'range': {
                        'sheetId':
                            self._get_sheet_id_by_name(
                                self.datasources_sheet_name),
                        'dimension':
                            'ROWS',
                        'startIndex':
                            row_idx - 1,  # 0-indexed here
                        'endIndex':
                            row_idx
                    }
                }
            }]
        }).execute()

    # Delete the associated data sheet
    sheet_name = f'data_{id}'
    try:
      self.sheets_service.spreadsheets().batchUpdate(
          spreadsheetId=self.config.spreadsheet_id,
          body={
              'requests': [{
                  'deleteSheet': {
                      'sheetId': self._get_sheet_id_by_name(sheet_name)
                  }
              }]
          }).execute()
    except Exception as e:
      logger.error(f'Error deleting data sheet: {e}')
      # Continue even if sheet deletion fails

    return True

  async def load_datasource_data(self, id: str) -> list[dict[str, Any]] | None:
    """Load data for a specific data source.

    Args:
      id: DataSource id.

    Returns:
      List of dictionaries containing the datasource data.

    Raises:
      ValueError: If datasource not found or any other error occurred.
    """
    datasource = await self.get_datasource_by_id(id, throw_if_not_found=True)

    # Load from the data sheet
    sheet_name = f'data_{id}'
    try:
      result = self.sheets_service.spreadsheets().values().get(
          spreadsheetId=self.config.spreadsheet_id,
          range=f'{sheet_name}!A:Z',
          valueRenderOption='FORMATTED_VALUE',
          dateTimeRenderOption='SERIAL_NUMBER').execute()

      values = result.get('values', [])
      logger.debug('Loaded data source, %s rows', len(values))
      if not values:
        return None

    except Exception as e:
      raise ValueError(
          f'Error loading data for data source with id={id}: {str(e)}') from e

    headers = values[0]
    data_rows = values[1:]

    # Transform into list of dictionaries (raw data)
    raw_data = []
    for row in data_rows:
      row_dict = {}
      for j, header in enumerate(headers):
        if j < len(row):
          val = row[j]
          if header in datasource.columns.metric_columns or header == datasource.columns.cost_column:
            try:
              val = float(val)
              if val.is_integer():
                val = int(val)
            except (ValueError, TypeError):
              val = row[j]
          if header == datasource.columns.date_column:
            val = date.fromisoformat(val)
          row_dict[header] = val
        else:
          row_dict[header] = None

      raw_data.append(row_dict)

    return raw_data

  async def initialize_master_spreadsheet(self):
    """Initialize the master spreadsheet if it doesn't exist."""
    # Check if we already have a master spreadsheet ID
    if self.config.spreadsheet_id:
      logger.debug('Using master spreadsheet %s', self.config.spreadsheet_id)
      return self.config.spreadsheet_id

    logger.info('Master Spreadsheet is not initialized, creating a new one')
    # Create a new spreadsheet
    spreadsheet = {
        'properties': {
            'title': 'GeoFlex Data'
        },
        'sheets': [{
            'properties': {
                'title': 'DataSources',
            }
        }]
    }

    spreadsheet = self.sheets_service.spreadsheets().create(
        body=spreadsheet).execute()

    master_spreadsheet_id = spreadsheet.get('spreadsheetId')
    sheet_id = spreadsheet['sheets'][0]['properties']['sheetId']

    # Add headers
    self.sheets_service.spreadsheets().values().update(
        spreadsheetId=master_spreadsheet_id,
        range='DataSources!A1:F1',
        valueInputOption='RAW',
        body={
            'values': [[
                'id', 'name', 'created_at', 'updated_at', 'source_link',
                'columns'
            ]]
        }).execute()

    # Format headers (make them bold)
    self.sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=master_spreadsheet_id,
        body={
            'requests': [{
                'repeatCell': {
                    'range': {
                        'sheetId': sheet_id,
                        'startRowIndex': 0,
                        'endRowIndex': 1
                    },
                    'cell': {
                        'userEnteredFormat': {
                            'textFormat': {
                                'bold': True
                            }
                        }
                    },
                    'fields': 'userEnteredFormat.textFormat.bold'
                }
            }]
        }).execute()

    self.config.spreadsheet_id = master_spreadsheet_id
    save_config(self.config)
    logger.info('Created master spreadsheet with id %s', master_spreadsheet_id)

    return self.config.spreadsheet_id

  async def share_master_spreadsheet(self, user_email: str):
    """Share the current master spreadsheet with a user.

    Args:
      user_email: a user's email
    """
    config = get_config()
    spreadsheet_id = config.spreadsheet_id

    if not spreadsheet_id:
      raise ValueError('No master spreadsheet configured')

    # Create permission for the user
    drive_service = self.get_drive_service()

    # Share with the user (using Drive API)
    permission = {'type': 'user', 'role': 'writer', 'emailAddress': user_email}

    drive_service.permissions().create(
        fileId=spreadsheet_id,
        body=permission,
        fields='id',
        sendNotificationEmail=False).execute()

  def get_drive_service(self):
    """Return a Google Drive API proxy for sharing operations."""
    scopes = ['https://www.googleapis.com/auth/drive']
    return build('drive', 'v3', credentials=get_credentials(scopes))

  def _get_sheet_id_by_name(self, sheet_name: str) -> int:
    """Get the sheet ID by its name."""
    sheet_metadata = self.sheets_service.spreadsheets().get(
        spreadsheetId=self.config.spreadsheet_id).execute()

    for sheet in sheet_metadata.get('sheets', []):
      if sheet.get('properties', {}).get('title') == sheet_name:
        return sheet.get('properties', {}).get('sheetId')
    raise ValueError(f'Sheet with name {sheet_name} was not found')

  async def preview_external_data(self, url: str) -> list[dict[str, Any]]:
    """Preview data from an external Google Sheets URL.

    Args:
      url: The Google Sheets URL to preview.

    Returns:
      List of dictionaries containing the preview data.

    Raises:
      ValueError: If URL is invalid or spreadsheet is not accessible.
    """
    # Extract spreadsheet ID from URL
    if 'spreadsheets/d/' not in url:
      raise ValueError('Invalid Google Sheets URL')

    parts = url.split('spreadsheets/d/')[1]
    spreadsheet_id = parts.split('/')[0]

    # Determine sheet name if specified
    sheet_name = None
    if '/edit#gid=' in url:
      # Extract gid from URL
      gid = url.split('#gid=')[1].split('&')[0]

      # Get spreadsheet metadata to find sheet name
      try:
        sheet_metadata = self.sheets_service.spreadsheets().get(
            spreadsheetId=spreadsheet_id).execute()

        # Find sheet with matching gid
        for sheet in sheet_metadata.get('sheets', []):
          if str(sheet.get('properties', {}).get('sheetId')) == gid:
            sheet_name = sheet.get('properties', {}).get('title')
            break

        if not sheet_name:
          logger.warning('Could not find sheet with gid %s, using first sheet',
                         gid)
      except Exception as e:
        logger.error('Error getting sheet metadata: %s, using first sheet', e)

    try:
      # Load from external spreadsheet
      result = self.sheets_service.spreadsheets().values().get(
          spreadsheetId=spreadsheet_id,
          range=f'{sheet_name}!A:Z' if sheet_name else 'A:Z').execute()

      values = result.get('values', [])
      if not values:
        return []

      # Transform into list of dictionaries
      headers = values[0]
      data_rows = values[1:]

      preview_data = []
      for row in data_rows:
        row_dict = {}
        for i, header in enumerate(headers):
          if i < len(row):
            row_dict[header] = row[i]
          else:
            row_dict[header] = None
        preview_data.append(row_dict)

      return preview_data

    except Exception as e:
      logger.error('Error loading external data: %s', e)
      if 'Unable to parse range' in str(e):
        raise ValueError('Invalid sheet name or range') from e
      elif 'Unable to parse spreadsheet' in str(e):
        raise ValueError('Invalid spreadsheet ID') from e
      elif 'Requested entity was not found' in str(e):
        raise ValueError('Spreadsheet not found') from e
      elif 'The caller does not have permission' in str(e):
        raise ValueError(
            f'No access to spreadsheet. Please share it with the service account {self.config.project_id}@appspot.gserviceaccount.com.'
        ) from e
      else:
        raise ValueError(f'Failed to load data: {str(e)}') from e
