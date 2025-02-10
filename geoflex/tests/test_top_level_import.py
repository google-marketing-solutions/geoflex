"""Tests for top level import."""

import types

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


def test_import_geoflex():
  """Tests that geoflex can be imported."""
  import geoflex  # pylint: disable=g-import-not-at-top

  assert isinstance(geoflex, types.ModuleType)
