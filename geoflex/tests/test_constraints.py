"""Tests for constraints."""

from geoflex import constraints
import pytest

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


def test_constraints_raise_exception_if_max_runtime_less_than_min_runtime():
  """Tests that geoflex can be imported."""
  with pytest.raises(ValueError):
    constraints.ExperimentDesignConstraints(
        experiment_type=constraints.ExperimentType.GO_DARK,
        max_runtime_weeks=1,
        min_runtime_weeks=2,
    )


def test_constraints_raise_exception_if_geos_in_both_fixed_treatment_and_fixed_control():
  """Tests that geoflex can be imported."""
  with pytest.raises(ValueError):
    constraints.ExperimentDesignConstraints(
        experiment_type=constraints.ExperimentType.GO_DARK,
        fixed_treatment_geos=["US", "UK"],
        fixed_control_geos=["US", "CA"],
    )


def test_constraints_can_be_created_with_valid_input():
  """Tests that geoflex can be imported."""
  constraints.ExperimentDesignConstraints(
      experiment_type=constraints.ExperimentType.GO_DARK,
      fixed_treatment_geos=["US", "UK"],
      fixed_control_geos=["CA", "AU"],
      max_runtime_weeks=4,
      min_runtime_weeks=2,
  )
