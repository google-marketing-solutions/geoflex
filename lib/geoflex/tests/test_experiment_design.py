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


@pytest.mark.parametrize(
    "treatment,control,exclude",
    [
        ([["US", "UK"], []], ["US", "AU"], ["FR"]),  # Treatment control overlap
        (
            [["US", "UK"], ["CA", "US"]],
            ["FR", "AU"],
            [],
        ),  # Treatment treatment overlap
        ([["US", "UK"], []], ["FR", "AU"], ["US"]),  # Treatment exclude overlap
        ([["US", "UK"], []], ["FR", "AU"], ["AU"]),  # Control exclude overlap
    ],
)
def test_geo_assignment_raises_exception_if_geos_in_multiple_groups(
    treatment, control, exclude
):
  with pytest.raises(ValueError):
    GeoAssignment(treatment=treatment, control=control, exclude=exclude)


def test_geo_assignment_raises_exception_if_treatment_geos_are_single_list():
  with pytest.raises(ValueError):
    # Treatment must be a list of lists to be able to have multiple treatment
    # arms.
    GeoAssignment(treatment=["US", "UK"], control=["US", "AU"])


@pytest.mark.parametrize(
    "invalid_args",
    [
        pytest.param(
            {"max_runtime_weeks": 1, "min_runtime_weeks": 2},
            id="min_runtime_weeks_greater_than_max_runtime_weeks",
        ),
        pytest.param(
            {"n_cells": 1},
            id="max_cells_less_than_2",
        ),
        pytest.param(
            {"n_geos_per_group_candidates": [[5, 1]], "n_cells": 3},
            id="n_geos_per_group_does_not_match_n_cells",
        ),
    ],
)
def test_constraints_raise_exception_inputs_are_invalid(invalid_args):
  with pytest.raises(ValueError):
    ExperimentDesignConstraints(
        experiment_type=ExperimentType.GO_DARK, **invalid_args
    )


@pytest.mark.parametrize(
    "valid_args",
    [
        {
            "fixed_geos": GeoAssignment(
                treatment=[["US", "UK"]], control=["CA", "AU"]
            ),
            "max_runtime_weeks": 4,
            "min_runtime_weeks": 2,
        },
        {},
        {"fixed_geos": None},
        {"n_geos_per_group_candidates": [[2, 2], [1, 5], None]},
        {"trimming_quantile_candidates": [0.0, 0.5]},
        {"n_cells": 3},
    ],
)
def test_constraints_can_be_created_with_valid_input(valid_args):
  ExperimentDesignConstraints(
      experiment_type=ExperimentType.GO_DARK, **valid_args
  )


@pytest.mark.parametrize(
    "methodology_parameters,expected_pretest_weeks",
    [
        ({"pretest_weeks": 4}, 4),
        ({}, 0),
    ],
)
def test_experiment_design_sets_pretest_weeks_correctly(
    methodology_parameters, expected_pretest_weeks
):
  design = ExperimentDesign(
      experiment_type=ExperimentType.GO_DARK,
      primary_metric="revenue",
      secondary_metrics=["conversions"],
      methodology="test_methodology",
      methodology_parameters=methodology_parameters,
      runtime_weeks=4,
      alpha=0.1,
      fixed_geos=None,
  )
  assert design.pretest_weeks == expected_pretest_weeks


def test_metric_names_must_be_unique():
  with pytest.raises(ValueError):
    ExperimentDesign(
        experiment_type=ExperimentType.GO_DARK,
        primary_metric="revenue",
        secondary_metrics=["revenue", "conversions"],
        methodology="test_methodology",
        runtime_weeks=4,
        alpha=0.1,
        fixed_geos=None,
    )
