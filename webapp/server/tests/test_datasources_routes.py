# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0330, g-bad-import-order, g-multiple-import, missing-module-docstring, missing-class-docstring, missing-function-docstring
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
import uuid
from server import app
from services.datasources import DataSourceService


@pytest.fixture
async def datasource_service(mock_sheets_api):
  """Create and initialize a DataSourceService."""
  service = DataSourceService()
  # Mock the initialize_master_spreadsheet method
  with patch.object(
      service,
      'initialize_master_spreadsheet',
      new=AsyncMock(return_value='test_spreadsheet_id')):
    await service.initialize_master_spreadsheet()
    return service


@pytest.fixture
def client(datasource_service):
  """Create a test client with initialized DataSourceService."""
  # Monkeypatch the app state to include our service
  app.state.datasource_service = datasource_service

  # Create the test client
  with TestClient(app) as client:
    yield client


@pytest.fixture
def mock_sheets_api():
  """Mock the Google Sheets API but keep the real DataSourceService."""
  #patch('services.datasources.get_credentials')
  with patch('services.datasources.build') as mock_build:
    # Setup mock sheets service
    mock_sheets = MagicMock()
    mock_spreadsheets = MagicMock()
    mock_values = MagicMock()

    # Configure the mock chain
    mock_build.return_value = mock_sheets
    mock_sheets.spreadsheets.return_value = mock_spreadsheets
    mock_spreadsheets.values.return_value = mock_values
    mock_spreadsheets.get = MagicMock()
    mock_spreadsheets.batchUpdate = MagicMock()

    # Setup execute chains for common methods
    mock_get_execute = MagicMock()
    mock_append_execute = MagicMock()
    mock_update_execute = MagicMock()
    mock_batchupdate_execute = MagicMock()
    mock_clear_execute = MagicMock()

    mock_values.get.return_value = mock_get_execute
    mock_values.append.return_value = mock_append_execute
    mock_values.update.return_value = mock_update_execute
    mock_values.clear.return_value = mock_clear_execute
    mock_spreadsheets.batchUpdate.return_value = mock_batchupdate_execute

    # Default empty responses
    mock_get_execute.execute.return_value = {'values': []}
    mock_append_execute.execute.return_value = {}
    mock_update_execute.execute.return_value = {}
    mock_batchupdate_execute.execute.return_value = {}
    mock_clear_execute.execute.return_value = {}

    # Also patch config
    with patch('services.datasources.get_config') as mock_config:
      config = MagicMock()
      config.spreadsheet_id = 'test_spreadsheet_id'
      mock_config.return_value = config

      yield {
          'build': mock_build,
          'sheets': mock_sheets,
          'spreadsheets': mock_spreadsheets,
          'values': mock_values,
          'get_execute': mock_get_execute,
          'append_execute': mock_append_execute,
          'update_execute': mock_update_execute,
          'batchupdate_execute': mock_batchupdate_execute
      }


@pytest.mark.asyncio
class TestDataSourceIntegrated:
  """Integrated tests for DataSource API that use the real service with mocked Google Sheets."""

  async def test_get_datasources(self, client, mock_sheets_api):
    # Arrange
    mock_datasource_data = [[
        'test-id-123', 'Test Data Source', '',
        datetime.now().isoformat(),
        datetime.now().isoformat(),
        'https://docs.google.com/spreadsheets/d/test',
        'Region:Date:Clicks,Conversions:Cost'
    ]]
    # Configure the mock to return our test data
    mock_sheets_api['get_execute'].execute.return_value = {
        'values': mock_datasource_data
    }

    # Act
    response = client.get('/api/datasources')

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]['id'] == 'test-id-123'
    assert data[0]['name'] == 'Test Data Source'
    assert data[0]['description'] == ''
    assert data[0]['columns']['geo_column'] == 'Region'
    assert data[0]['columns']['metric_columns'] == ['Clicks', 'Conversions']
    # Verify Google Sheets API was called
    mock_sheets_api['values'].get.assert_called()

  async def test_create_datasource(self, client, mock_sheets_api):
    # Arrange
    mock_sheets_api['append_execute'].execute.return_value = {}
    ds_id = '12345678-1234-5678-1234-567812345678'
    with patch('uuid.uuid4', return_value=uuid.UUID(ds_id)):
      request_data = {
          'name':
              'Data',
          'id':
              '',
          'description':
              '',
          'source_link':
              'https://docs.google.com/spreadsheets/d/new',
          'columns': {
              'geo_column': 'City',
              'date_column': 'Week',
              'metric_columns': ['Impressions', 'Clicks', 'Conversions'],
              'cost_column': 'Spend'
          },
          'data': [{
              'City': 'Moscow',
              'Week': 1,
              'Impressions': 0,
              'Clicks': 0,
              'Conversions': 0,
              'Spend': 0,
          }]
      }

      # Act
      response = client.post('/api/datasources', json=request_data)

      # Assert
      assert response.status_code == 200
      data = response.json()
      assert data['id'] == ds_id
      assert data['name'] == 'Data'
      assert data['columns']['geo_column'] == 'City'
      assert data['columns']['metric_columns'] == [
          'Impressions', 'Clicks', 'Conversions'
      ]
      # Verify Google Sheets API was called
      mock_sheets_api['values'].append.assert_called_once()

  async def test_update_datasource(self, client, mock_sheets_api):
    # Arrange
    ds_id = 'existing-id-123'
    existing_ds_time = datetime.now().isoformat()
    mock_sheets_api['get_execute'].execute.side_effect = [{
        'values': [[
            ds_id, 'Original Name', '', existing_ds_time, existing_ds_time,
            'https://docs.google.com/spreadsheets/d/original',
            'Region:Date:Clicks,Impressions:Cost'
        ]]
    }]
    mock_sheets_api['update_execute'].execute.return_value = {}
    request_data = {
        'id': ds_id,
        'name': 'Updated Campaign Name',
        'description': '',
        'source_link': 'https://docs.google.com/spreadsheets/d/updated',
        'columns': {
            'geo_column': 'Country',
            'date_column': 'Month',
            'metric_columns': ['Revenue', 'Conversions', 'CPA'],
            'cost_column': 'Budget'
        }
    }

    # Act
    response = client.put('/api/datasources/existing-id-123', json=request_data)

    # Assert
    data = response.json()
    print(data)
    assert response.status_code == 200
    assert data['id'] == 'existing-id-123'
    assert data['name'] == 'Updated Campaign Name'
    assert data['columns']['geo_column'] == 'Country'
    assert data['columns']['metric_columns'] == [
        'Revenue', 'Conversions', 'CPA'
    ]
    # Verify Google Sheets API was called
    assert mock_sheets_api['values'].get.call_count >= 1
    mock_sheets_api['values'].update.assert_called_once()

  async def test_delete_datasource(self, client, mock_sheets_api):
    # Arrange
    existing_ds_time = datetime.now().isoformat()
    mock_sheets_api['get_execute'].execute.side_effect = [
        # First for get_datasource_by_id
        {
            'values': [[
                'delete-test-id', 'Datasource To Delete', '', existing_ds_time,
                existing_ds_time,
                'https://docs.google.com/spreadsheets/d/todelete',
                'Region:Date:Metric1,Metric2:Cost'
            ]]
        },
        # Second for find row
        {
            'values': [['delete-test-id']]
        }
    ]
    # Configure spreadsheets.get for _get_sheet_id_by_name
    spreadsheet_metadata_mock = MagicMock()
    spreadsheet_metadata_mock.execute.return_value = {
        'sheets': [{
            'properties': {
                'title': 'DataSources',
                'sheetId': 0
            }
        }, {
            'properties': {
                'title': 'data_delete-test-id',
                'sheetId': 1
            }
        }]
    }
    mock_sheets_api['spreadsheets'].get.return_value = spreadsheet_metadata_mock

    # Act
    response = client.delete('/api/datasources/delete-test-id')

    # Assert
    assert response.status_code >= 200 and response.status_code < 300
    # Verify Google Sheets API was called
    mock_sheets_api['spreadsheets'].batchUpdate.assert_called()

  async def test_load_datasource_data(self, client, mock_sheets_api):
    # Arrange
    existing_ds_time = datetime.now().isoformat()
    mock_sheets_api['get_execute'].execute.side_effect = [
        # First call - for get_datasource_by_id
        {
            'values': [[
                'data-test-id', 'Test Data Loading', '', existing_ds_time,
                existing_ds_time,
                'https://docs.google.com/spreadsheets/d/testdata',
                'City:Week:Sales,Returns:Advertising'
            ]]
        },
        # Second call - for loading data
        {
            'values': [['City', 'Week', 'Sales', 'Returns', 'Advertising'],
                       ['New York', '2023-W01', '5000', '250', '1200'],
                       ['Chicago', '2023-W01', '3800', '190', '950'],
                       ['Los Angeles', '2023-W01', '4500', '225', '1100']]
        }
    ]

    # Act
    response = client.get('/api/datasources/data-test-id/data')

    # Assert
    assert response.status_code == 200
    data = response.json()
    # we expect to get list of dictionaries
    assert len(data) == 3
    assert 'City' in data[0]
    assert 'Week' in data[0]
    assert 'Sales' in data[0]
    assert 'Returns' in data[0]
    assert 'Advertising' in data[0]
    assert 'New York' == data[0]['City']
    assert '2023-W01' == data[1]['Week']
    assert 4500 == data[2]['Sales']
    # Verify Google Sheets API was called
    assert mock_sheets_api['values'].get.call_count >= 2
