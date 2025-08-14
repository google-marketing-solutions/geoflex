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

"""TBRMM specific tests to be run on top of the standard tests.

Focus is on ensuring the TBRMM wrapper is working. The original library is
tested in its own unit tests and the wrapper is dependent on the original
library working correctly.
"""

import copy
import logging

import geoflex.data
import geoflex.experiment_design
import geoflex.exploration_spec
import geoflex.methodology
import geoflex.methodology.tbrmm
import geoflex.metrics
import numpy as np
import pandas as pd
import pytest


TBRMM = geoflex.methodology.tbrmm.TBRMM
ExperimentDesignExplorationSpec = (
    geoflex.exploration_spec.ExperimentDesignExplorationSpec
)
GeoAssignment = geoflex.experiment_design.GeoAssignment
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesign = geoflex.experiment_design.ExperimentDesign
GeoEligibility = geoflex.experiment_design.GeoEligibility
ExperimentBudget = geoflex.experiment_design.ExperimentBudget
ExperimentBudgetType = geoflex.experiment_design.ExperimentBudgetType

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


@pytest.fixture(name="performance_data")
def performance_data_fixture():
  """Fixture for historical data suitable for TBRMM."""
  rng = np.random.default_rng(seed=42)
  data = pd.DataFrame({
      "geo_id": np.repeat([f"geo_{i}" for i in range(20)], 100),
      "date": pd.date_range(start="2024-01-01", periods=100).tolist() * 20,
      "revenue": rng.random(size=2000) + 5,
      "cost": rng.random(size=2000) + 5,
      "conversions": rng.random(size=2000) + 5,
  })
  data["date"] = data["date"].dt.strftime("%Y-%m-%d")
  return GeoPerformanceDataset(data=data)


@pytest.fixture(name="few_geos_performance_data")
def few_geos_performance_data_fixture():
  """Fixture for historical data with few geos for TBRMM exhaustive search."""
  rng = np.random.default_rng(seed=42)
  data = pd.DataFrame({
      "geo_id": np.repeat([f"geo_{i}" for i in range(6)], 100),
      "date": pd.date_range(start="2024-01-01", periods=100).tolist() * 6,
      "revenue": rng.random(size=600) + 5,
      "cost": rng.random(size=600) + 5,
      "conversions": rng.random(size=600) + 5,
  })
  data["date"] = data["date"].dt.strftime("%Y-%m-%d")
  return GeoPerformanceDataset(data=data)


@pytest.fixture(name="base_design_dict")
def base_design_dict_fixture():
  """Returns a dictionary for a basic, valid ExperimentDesign for TBRMM."""
  return {
      "primary_metric": "revenue",
      "methodology": "TBRMM",
      "runtime_weeks": 2,
      "n_cells": 2,
      "methodology_parameters": {"pretest_weeks": 4},
  }


@pytest.mark.parametrize(
    "design, expected_eligibility",
    [
        (
            ExperimentDesign(
                primary_metric="revenue",
                methodology="TBRMM",
                runtime_weeks=2,
                n_cells=2,
                methodology_parameters={"pretest_weeks": 4},
            ),
            True,
        ),
        (
            ExperimentDesign(
                primary_metric="revenue",
                methodology="TBRMM",
                runtime_weeks=2,
                n_cells=2,
                methodology_parameters={"pretest_weeks": 4},
                geo_eligibility=GeoEligibility(
                    control={"geo_0"}, treatment=[{"geo_1"}]
                ),
            ),
            True,
        ),
        (
            ExperimentDesign(
                primary_metric="revenue",
                methodology="TBRMM",
                runtime_weeks=2,
                n_cells=3,
                methodology_parameters={"pretest_weeks": 4},
            ),
            False,  # n_cells must be 2
        ),
    ],
)


def test_tbrmm_eligibility(design, expected_eligibility, performance_data):
  assert (
      TBRMM().is_eligible_for_design_and_data(design, performance_data)
      == expected_eligibility
  )


@pytest.mark.parametrize(
    "design_params, expected_geo_counts, should_raise_error",
    [
        (
            {
                "methodology_parameters": {
                    "treatment_geos_range": (1, 2),
                }
            },
            {"min_treatment": 1, "max_treatment": 2},
            False,
        ),
        (
            {
                "methodology_parameters": {
                    "control_geos_range": (1, 5),
                }
            },
            {"min_control": 1, "max_control": 5},
            False,
        ),
    ],
)


def test_tbrmm_assign_geos_with_constraints(
    design_params,
    expected_geo_counts,
    should_raise_error,
    performance_data,
    base_design_dict,
):
  # Deep merge the design_params into the base_design_dict
  design_dict = copy.deepcopy(base_design_dict)
  if "methodology_parameters" in design_params:
    design_dict["methodology_parameters"].update(
        design_params.pop("methodology_parameters")
    )
  design_dict.update(design_params)
  experiment_design = ExperimentDesign(**design_dict)
  if should_raise_error:
    with pytest.raises(
        (ValueError, RuntimeError), match="returned no suitable designs"
    ):
      TBRMM().assign_geos(experiment_design, performance_data)
  else:
    geo_assignment, _ = TBRMM().assign_geos(
        experiment_design, performance_data
    )
    assert isinstance(geo_assignment, GeoAssignment)
    if expected_geo_counts:
      if "max_treatment" in expected_geo_counts:
        assert (
            len(geo_assignment.treatment[0])
            <= expected_geo_counts["max_treatment"]
        )
      if "min_treatment" in expected_geo_counts:
        assert (
            len(geo_assignment.treatment[0])
            >= expected_geo_counts["min_treatment"]
        )
      if "max_control" in expected_geo_counts:
        assert len(geo_assignment.control) <= expected_geo_counts["max_control"]
      if "min_control" in expected_geo_counts:
        assert len(geo_assignment.control) >= expected_geo_counts["min_control"]


@pytest.mark.parametrize(
    "geo_eligibility, assertions",
    [
        (
            GeoEligibility(
                control={"geo_0", "geo_1"},
                treatment=[{"geo_1"}],
                flexible=True,
            ),
            [
                # geo_0 is eligible for control but may end up in exclude
                # only assert that it is not in treatment
                lambda ga: "geo_0" not in ga.treatment[0],
            ],
        ),
        (
            GeoEligibility(
                control={"geo_0", "geo_1"},
                treatment=[{"geo_2", "geo_3"}],
                flexible=False,
            ),
            [
                # inflexible eligibility means the assignment must align with
                # specified eligibility lists
                lambda ga: ga.control.issubset({"geo_0", "geo_1"}),
                lambda ga: ga.treatment[0].issubset({"geo_2", "geo_3"}),
                # geos not in eligibility lists must be in exclude
                lambda ga: "geo_5" in ga.exclude,
                lambda ga: "geo_5" not in (ga.control | ga.treatment[0]),
            ],
        ),
    ],
)


def test_tbrmm_assign_geos_with_geo_eligibility(
    geo_eligibility, assertions, performance_data, base_design_dict
):
  design_dict = copy.deepcopy(base_design_dict)
  design_dict["geo_eligibility"] = geo_eligibility
  experiment_design = ExperimentDesign(**design_dict)
  geo_assignment, _ = TBRMM().assign_geos(experiment_design, performance_data)
  for assertion in assertions:
    assert assertion(geo_assignment)


@pytest.mark.parametrize(
    "budget, cost_column, expected_log_fragment",
    [
        (
            ExperimentBudget(
                budget_type=ExperimentBudgetType.TOTAL_BUDGET, value=500.0
            ),
            "cost",
            """Mapped GeoFleX budget type 'TOTAL_BUDGET' with value 500.0
            (total) to original library budget_range (0.0, 500.0)""",
        ),
        (
            ExperimentBudget(
                budget_type=ExperimentBudgetType.DAILY_BUDGET, value=50.0
            ),
            "cost",
            """Mapped GeoFleX budget type 'DAILY_BUDGET' with value 50.0
            (daily) to original library budget_range (0.0, 700.0)
            (total over 2 runtime weeks)""",
        ),
        (
            ExperimentBudget(
                budget_type=ExperimentBudgetType.TOTAL_BUDGET, value=500.0
            ),
            None,
            """A budget is specified for the experiment, but the 'cost_column'
            is not set""",
        ),
    ],
)


def test_tbrmm_assign_geos_with_budget_constraint(
    budget,
    cost_column,
    expected_log_fragment,
    performance_data,
    base_design_dict,
    caplog,
):
  design_dict = copy.deepcopy(base_design_dict)
  design_dict["experiment_budget"] = budget
  if cost_column:
    design_dict["methodology_parameters"]["cost_column"] = cost_column

  experiment_design = ExperimentDesign(**design_dict)

  with caplog.at_level(logging.INFO):
    # not checking the output, just that the parameters are passed
    # search may or may not find a suitable design
    try:
      TBRMM().assign_geos(experiment_design, performance_data)
    except (ValueError, RuntimeError):
      pass

  assert expected_log_fragment in caplog.text


@pytest.mark.parametrize(
    "fixture_name, expected_log_fragment",
    [
        ("performance_data", "Using greedy search"),
        ("few_geos_performance_data", "Attempting exhaustive search"),
    ],
)


def test_tbrmm_assign_geos_search_path(
    fixture_name, expected_log_fragment, request, caplog, base_design_dict
):
  # test that search path is correct based on the number of geos
  performance_data = request.getfixturevalue(fixture_name)
  experiment_design = ExperimentDesign(**copy.deepcopy(base_design_dict))
  with caplog.at_level(logging.INFO):
    geo_assignment, _ = TBRMM().assign_geos(
        experiment_design, performance_data
    )
  assert isinstance(geo_assignment, GeoAssignment)
  assert geo_assignment.control
  assert len(geo_assignment.treatment) == 1
  assert geo_assignment.treatment[0]
  assert not (geo_assignment.control & geo_assignment.treatment[0])
  assert expected_log_fragment in caplog.text


def test_tbrmm_analyze_experiment(performance_data, base_design_dict):
  design_dict = copy.deepcopy(base_design_dict)
  design_dict["secondary_metrics"] = ["conversions"]
  experiment_design = ExperimentDesign(**design_dict)
  # A plausible geo assignment for the analysis step.
  geos = list(performance_data.geos)
  experiment_design.geo_assignment = GeoAssignment(
      control=set(geos[:10]), treatment=[set(geos[10:])]
  )
  analysis_results, _ = TBRMM().analyze_experiment(
      runtime_data=performance_data,
      experiment_design=experiment_design,
      experiment_start_date="2024-03-01",
      experiment_end_date="2024-03-15",
  )
  assert isinstance(analysis_results, pd.DataFrame)
  assert not analysis_results.empty
