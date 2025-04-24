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
"""Model classes for data sources."""

# pylint: disable=C0330, g-bad-import-order, g-multiple-import, g-importing-member
from pydantic import BaseModel
from typing import Any
from datetime import datetime


class ColumnSchema(BaseModel):
  """Data Source column configuration."""
  geo_column: str
  date_column: str
  metric_columns: list[str]
  cost_column: str | None = None


class DataSource(BaseModel):
  """Data Source metadata and optionally data."""
  id: str | None
  name: str
  description: str | None = None
  columns: ColumnSchema
  created_at: datetime | None
  updated_at: datetime | None
  source_link: str
  data: list[dict[str, Any]] | None = None
