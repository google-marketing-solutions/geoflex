"""Tests for the experiment design module."""

import geoflex.experiment_design
import pytest

ExperimentDesignConstraints = (
    geoflex.experiment_design.ExperimentDesignConstraints
)
ExperimentType = geoflex.experiment_design.ExperimentType
GeoAssignment = geoflex.experiment_design.GeoAssignment
ExperimentDesign = geoflex.experiment_design.ExperimentDesign

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


def test_constraints_raise_exception_if_max_runtime_less_than_min_runtime():
  with pytest.raises(ValueError):
    ExperimentDesignConstraints(
        experiment_type=ExperimentType.GO_DARK,
        max_runtime_weeks=1,
        min_runtime_weeks=2,
    )


def test_constraints_raise_exception_if_geos_in_both_fixed_treatment_and_fixed_control():
  with pytest.raises(ValueError):
    ExperimentDesignConstraints(
        experiment_type=ExperimentType.GO_DARK,
        fixed_geos=GeoAssignment(treatment=["US", "UK"], control=["US", "AU"]),
    )


def test_constraints_can_be_created_with_valid_input():
  ExperimentDesignConstraints(
      experiment_type=ExperimentType.GO_DARK,
      fixed_geos=GeoAssignment(treatment=["US", "UK"], control=["CA", "AU"]),
      max_runtime_weeks=4,
      min_runtime_weeks=2,
  )


def test_geo_assignment_evaluates_to_true_if_geos_are_not_empty():
  assert GeoAssignment(control=["US", "UK"], treatment=[])
  assert GeoAssignment(control=[], treatment=["US", "UK"])
  assert GeoAssignment(control=["US", "UK"], treatment=["CA", "AU"])
  assert not GeoAssignment(control=[], treatment=[])
