# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for top level import."""

import logging
import types
import pytest

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


def test_import_geoflex():
  """Tests that geoflex can be imported."""
  import geoflex  # pylint: disable=g-import-not-at-top

  assert isinstance(geoflex, types.ModuleType)


def test_temporay_log_level():
  """Tests that the log level is changed and restored."""
  import geoflex  # pylint: disable=g-import-not-at-top

  original_level = geoflex.logger.getEffectiveLevel()
  target_level = logging.DEBUG
  if original_level == logging.DEBUG:
    target_level = logging.INFO

  assert geoflex.logger.getEffectiveLevel() == original_level
  assert geoflex.logger.getEffectiveLevel() != target_level

  with geoflex.temporary_log_level(target_level):
    assert geoflex.logger.getEffectiveLevel() == target_level

  assert geoflex.logger.getEffectiveLevel() == original_level


def test_temporary_log_level_with_exception():
  """Tests that the log level is restored after an exception."""
  import geoflex  # pylint: disable=g-import-not-at-top

  original_level = geoflex.logger.getEffectiveLevel()
  target_level = logging.DEBUG
  if original_level == logging.DEBUG:
    target_level = logging.INFO

  assert geoflex.logger.getEffectiveLevel() == original_level
  assert geoflex.logger.getEffectiveLevel() != target_level

  try:
    with pytest.raises(ValueError):
      with geoflex.temporary_log_level(target_level):
        assert geoflex.logger.getEffectiveLevel() == target_level
        raise ValueError("test exception")
  finally:
    assert geoflex.logger.getEffectiveLevel() == original_level
