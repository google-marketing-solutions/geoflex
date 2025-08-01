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
"""Service for storing and retrieving experiment designs."""

# pylint: disable=C0330, g-bad-import-order, g-multiple-import, g-importing-member
import json
import smart_open
import gcsfs
from typing import Any
from config import Config
from logger import logger


class DesignStorageService:
  """Handles the storage and retrieval of experiment designs from GCS."""

  def __init__(self, config: Config):
    """Initializes the DesignStorageService.

    Args:
      config: The application configuration.
    """
    gcs_bucket_name = config.gcs_bucket_name
    if not gcs_bucket_name:
      gcs_bucket_name = f"{config.project_id}/geoflex/designs"
    self.bucket_path = f"gs://{gcs_bucket_name}"

  async def list_designs(self) -> list[dict[str, Any]]:
    """Lists all available designs and their content from the GCS bucket.

    Returns:
      A list of design objects.
    """
    designs = []
    try:
      fs = gcsfs.GCSFileSystem(listings_expiry_time=0)
      files = fs.ls(self.bucket_path, detail=False)
      logger.debug('Loaded design files: %s', files)
      for file_path in files:
        with smart_open.open(f"gs://{file_path}", 'r') as design_file:
          designs.append(json.load(design_file))
    except Exception as e:
      logger.error(f"Error listing designs from GCS {self.bucket_path}: {e}")
    return designs

  async def get_design(self, design_name: str) -> dict[str, Any] | None:
    """Retrieves a specific design file from GCS.

    Args:
      design_name: The name of the design file to retrieve.

    Returns:
      The design content as a dictionary, or None if not found.
    """
    file_path = f"{self.bucket_path}/{design_name}"
    try:
      with smart_open.open(file_path, 'r') as f:
        return json.load(f)
    except Exception as e:
      logger.error(f"Error reading design '{file_path}' from GCS: {e}")
      return None

  async def save_design(self, design_name: str, design_data: dict[str,
                                                                  Any]) -> bool:
    """Saves a design file to GCS.

    Args:
      design_name: The name of the design file to save.
      design_data: The design content to save.

    Returns:
      True if successful, False otherwise.
    """
    file_path = f"{self.bucket_path}/{design_name}"
    try:
      with smart_open.open(file_path, 'w') as f:
        json.dump(design_data, f, indent=2)
      logger.info(f"Successfully saved design '{file_path}' to GCS.")
      return True
    except Exception as e:
      logger.error(f"Error saving design '{file_path}' to GCS: {e}")
      return False

  async def delete_design(self, design_name: str) -> bool:
    """Deletes a design file from GCS.

    Args:
      design_name: The name of the design file to delete.

    Returns:
      True if successful, False otherwise.
    """
    file_path = f"{self.bucket_path}/{design_name}"
    try:
      fs = gcsfs.GCSFileSystem()
      fs.rm(file_path)
      logger.info(f"Successfully deleted design '{file_path}' from GCS.")
      return True
    except Exception as e:
      logger.error(f"Error deleting design '{file_path}' from GCS: {e}")
      return False
