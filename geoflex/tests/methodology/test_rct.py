"""Tests for the RCT methodology module."""

import geoflex.experiment_design
import geoflex.methodology.rct
import pytest

RCT = geoflex.methodology.rct.RCT
ExperimentDesignConstraints = (
    geoflex.experiment_design.ExperimentDesignConstraints
)
ExperimentType = geoflex.experiment_design.ExperimentType
GeoAssignment = geoflex.experiment_design.GeoAssignment

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


@pytest.mark.parametrize(
    "fixed_control_geos,fixed_treatment_geos,expected_is_eligible",
    [
        (["US", "UK"], ["CA", "AU"], False),
        (["US", "UK"], [], False),
        ([], ["CA", "AU"], False),
        ([], [], True),
    ],
)
def test_rct_is_eligible_for_constraints(
    fixed_control_geos, fixed_treatment_geos, expected_is_eligible
):
  constraints = ExperimentDesignConstraints(
      experiment_type=ExperimentType.GO_DARK,
      max_runtime_weeks=4,
      min_runtime_weeks=2,
      fixed_geos=GeoAssignment(
          control=fixed_control_geos, treatment=fixed_treatment_geos
      ),
  )
  assert RCT().is_eligible_for_constraints(constraints) == expected_is_eligible
