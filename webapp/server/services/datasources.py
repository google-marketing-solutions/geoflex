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

# pylint: disable=C0330, g-bad-import-order, g-multiple-import
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import google.auth
from google.auth import credentials
from googleapiclient.discovery import build
from logger import logger
from config import get_config, save_config


def get_credentials(scopes) -> credentials.Credentials:
  creds, _ = google.auth.default(scopes)
  return creds


class DataSourceService:

  def __init__(self):
    # Initialize Google Sheets API
    self._init_sheets_api()

    self.config = get_config(fail_if_not_exists=False)
    self.datasources_sheet_name = 'DataSources'


  def _init_sheets_api(self):
    """Initialize Google Sheets API client."""
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    credentials = get_credentials(SCOPES)
    self.sheets_service = build('sheets', 'v4', credentials=credentials)
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
        range='DataSources!A1:G1',
        valueInputOption='RAW',
        body={
            'values': [[
                'id', 'name', 'created_at', 'updated_at', 'type', 'source_link',
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
    """Share the current master spreadsheet with a user."""
    config = get_config()
    spreadsheet_id = config.spreadsheet_id

    if not spreadsheet_id:
      raise ValueError("No master spreadsheet configured")

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
    """Create a Google Drive API service for sharing operations"""
    SCOPES = ['https://www.googleapis.com/auth/drive']
    credentials = get_credentials(SCOPES)
    return build('drive', 'v3', credentials=credentials)
