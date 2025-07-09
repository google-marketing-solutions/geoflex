"""Standard tests that should be run for all methodologies.

You can also include specific tests in their own files for each methodology,
in addition to the ones here.
"""

import geoflex.data
import geoflex.experiment_design
import geoflex.exploration_spec
import geoflex.methodology
import geoflex.metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


TestingMethodology = geoflex.methodology.testing_methodology.TestingMethodology
ExperimentDesignExplorationSpec = (
    geoflex.exploration_spec.ExperimentDesignExplorationSpec
)
GeoAssignment = geoflex.experiment_design.GeoAssignment
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesign = geoflex.experiment_design.ExperimentDesign
GeoEligibility = geoflex.experiment_design.GeoEligibility
ExperimentBudget = geoflex.experiment_design.ExperimentBudget
ExperimentBudgetType = geoflex.experiment_design.ExperimentBudgetType
CellVolumeConstraint = geoflex.experiment_design.CellVolumeConstraint
CellVolumeConstraintType = geoflex.experiment_design.CellVolumeConstraintType

assign_geos = geoflex.methodology.assign_geos
analyze_experiment = geoflex.methodology.analyze_experiment
design_is_eligible_for_data = geoflex.methodology.design_is_eligible_for_data

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


@pytest.fixture(name="methodology", scope="function")
def methodology_fixture(request):
  """Fixture to get the methodology name from parameterized tests."""
  return request.param


@pytest.fixture(name="performance_data", scope="function")
def performance_data_fixture(request):
  """Fixture for the performance data."""
  rng = np.random.default_rng(seed=42)

  if request.param == "basic_data":
    data = pd.DataFrame({
        "geo_id": [f"geo_{i}" for i in range(20) for _ in range(100)],  # pylint: disable=g-complex-comprehension
        "date": pd.date_range(start="2024-01-01", periods=100).tolist() * 20,
        "revenue": rng.random(size=2000) + 5,
        "cost": rng.random(size=2000) + 5,
        "conversions": rng.random(size=2000) + 5,
    })
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")
    return GeoPerformanceDataset(data=data)
  elif request.param == "gbr_wls_problematic_data":
    data = pd.DataFrame({
        "geo_id": [f"geo_{i}" for i in range(20) for _ in range(100)],  # pylint: disable=g-complex-comprehension
        "date": pd.date_range(start="2024-01-01", periods=100).tolist() * 20,
        "revenue": rng.random(size=2000) + 5,
        "cost": rng.random(size=2000) + 5,
        "conversions": rng.random(size=2000) + 5,
    })

    data.loc[data["geo_id"] == "geo_0", "cost"] = 0  # Add some zeros
    data.loc[data["geo_id"] == "geo_0", "revenue"] = 0  # Add some zeros
    data.loc[data["geo_id"] == "geo_0", "conversions"] = 0  # Add some zeros
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")
    return GeoPerformanceDataset(data=data)
  elif request.param == "gbr_few_geos_data":
    # Only 6 geos, for a 2-cell design, this is 3 geos/cell < 4
    data = pd.DataFrame({
        "geo_id": [f"geo_{i}" for i in range(6) for _ in range(100)],  # pylint: disable=g-complex-comprehension
        "date": pd.date_range(start="2024-01-01", periods=100).tolist() * 6,
        "revenue": rng.random(size=600) + 5,
        "cost": rng.random(size=600) + 5,
        "conversions": rng.random(size=600) + 5,
    })
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")
    return GeoPerformanceDataset(data=data)
  else:
    raise ValueError(f"Unknown performance data: {request.param}")


@pytest.fixture(name="experiment_design", scope="function")
def experiment_design_fixture(request, methodology):
  """Fixture for the performance data."""
  if request.param == "unconstrained_ab_test":
    return ExperimentDesign(
        primary_metric="revenue",
        secondary_metrics=["conversions"],
        methodology=methodology,
        runtime_weeks=2,
        n_cells=2,
    )
  elif request.param == "ab_test_with_excluded_geos":
    return ExperimentDesign(
        primary_metric="revenue",
        secondary_metrics=["conversions"],
        methodology=methodology,
        runtime_weeks=2,
        n_cells=2,
        geo_eligibility=GeoEligibility(exclude={"geo_0", "geo_1"}),
    )
  elif request.param == "ab_test_with_fixed_geos":
    return ExperimentDesign(
        primary_metric="revenue",
        secondary_metrics=["conversions"],
        methodology=methodology,
        runtime_weeks=2,
        n_cells=2,
        geo_eligibility=GeoEligibility(
            treatment=[{"geo_2", "geo_3"}],
            control={"geo_6", "geo_7"},
        ),
    )
  elif request.param == "gbr_wls_design":
    assert methodology == "GBR"
    return ExperimentDesign(
        primary_metric="revenue",
        secondary_metrics=["conversions"],
        methodology=methodology,  # Should be "GBR" when parameterized
        runtime_weeks=2,
        n_cells=2,
        methodology_parameters={"linear_model_type": "wls"},
    )
  elif request.param == "gbr_robust_ols_design":
    assert methodology == "GBR"
    return ExperimentDesign(
        primary_metric="revenue",
        secondary_metrics=["conversions"],
        methodology=methodology,  # Should be "GBR" when parameterized
        runtime_weeks=2,
        n_cells=2,
        methodology_parameters={"linear_model_type": "robust_ols"},
    )
  elif request.param == "multicell_ab_test":
    return ExperimentDesign(
        primary_metric="revenue",
        secondary_metrics=["conversions"],
        methodology=methodology,
        runtime_weeks=2,
        n_cells=3,  # Multi-cell
    )
  elif request.param == "cost_metrics_test":
    return ExperimentDesign(
        primary_metric=geoflex.metrics.iROAS(),  # Cost-based metric
        secondary_metrics=[geoflex.metrics.CPiA()],  # Cost-based metric
        experiment_budget=ExperimentBudget(
            budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
            value=-1.0,
        ),
        methodology=methodology,
        runtime_weeks=2,
        n_cells=2,
    )
  elif request.param == "max_geos_constraint_test":
    return ExperimentDesign(
        primary_metric="revenue",
        secondary_metrics=["conversions"],
        methodology=methodology,
        runtime_weeks=2,
        n_cells=2,
        cell_volume_constraint=CellVolumeConstraint(
            constraint_type=CellVolumeConstraintType.MAX_GEOS,
            values=[None, 10],
        ),
    )
  elif request.param == "max_revenue_pct_constraint_test":
    return ExperimentDesign(
        primary_metric="revenue",
        secondary_metrics=["conversions"],
        methodology=methodology,
        runtime_weeks=2,
        n_cells=2,
        cell_volume_constraint=CellVolumeConstraint(
            constraint_type=CellVolumeConstraintType.MAX_PERCENTAGE_OF_METRIC,
            values=[None, 0.5],
            metric_column="revenue",
        ),
    )
  else:
    raise ValueError(f"Unknown experiment design: {request.param}")


VALID_COMBINATIONS = [
    ("GBR", "unconstrained_ab_test", "basic_data"),
    ("GBR", "ab_test_with_excluded_geos", "basic_data"),
    ("GBR", "gbr_wls_design", "basic_data"),
    ("GBR", "gbr_robust_ols_design", "basic_data"),
    ("GBR", "gbr_robust_ols_design", "gbr_wls_problematic_data"),
    ("GBR", "multicell_ab_test", "basic_data"),
    ("GBR", "cost_metrics_test", "basic_data"),
    ("GBR", "max_geos_constraint_test", "basic_data"),
    ("GBR", "max_revenue_pct_constraint_test", "basic_data"),
]
INVALID_COMBINATIONS = [
    (
        "GBR",
        "ab_test_with_fixed_geos",
        "basic_data",
    ),  # Fixed geos are not supported for GBR
    (
        "GBR",
        "gbr_wls_design",
        "gbr_wls_problematic_data",
    ),  # WLS fails with problematic data
    (
        "GBR",
        "unconstrained_ab_test",
        "gbr_few_geos_data",
    ),  # Default GBR (WLS) fails with too few geos
    (
        "GBR",
        "gbr_wls_design",
        "gbr_few_geos_data",
    ),  # Explicit WLS GBR fails with too few geos
    (
        "GBR",
        "gbr_robust_ols_design",
        "gbr_few_geos_data",
    ),  # Robust OLS GBR fails with too few geos
]


@pytest.mark.parametrize(
    "methodology,experiment_design,performance_data",
    VALID_COMBINATIONS,
    indirect=True,
)
def test_design_is_eligible_for_data_returns_true_for_valid_combinations(
    methodology, experiment_design, performance_data
):
  assert experiment_design.methodology == methodology
  assert design_is_eligible_for_data(experiment_design, performance_data)


@pytest.mark.parametrize(
    "methodology,experiment_design,performance_data",
    INVALID_COMBINATIONS,
    indirect=True,
)
def test_design_is_eligible_for_data_returns_false_for_invalid_combinations(
    methodology, experiment_design, performance_data
):
  assert experiment_design.methodology == methodology
  assert not design_is_eligible_for_data(experiment_design, performance_data)


@pytest.mark.parametrize(
    "methodology,experiment_design,performance_data",
    VALID_COMBINATIONS,
    indirect=True,
)
def test_assign_geos_returns_expected_data_type(
    methodology, experiment_design, performance_data
):
  assert experiment_design.methodology == methodology
  geo_assignment = assign_geos(experiment_design, performance_data)
  assert isinstance(geo_assignment, GeoAssignment)

  # experiment_design: ExperimentDesign,
  # runtime_data: GeoPerformanceDataset,
  # experiment_start_date: str,
  # experiment_end_date: str | None = None,
  # pretest_period_end_date: str | None = None,
  # return_intermediate_data: bool = False,
  # with_deep_dive_plots: bool = False,


@pytest.mark.parametrize("with_deep_dive_plots", [True, False])
@pytest.mark.parametrize("return_intermediate_data", [True, False])
@pytest.mark.parametrize(
    "methodology,experiment_design,performance_data",
    VALID_COMBINATIONS,
    indirect=True,
)
def test_analyze_experiment_returns_expected_data_type(
    methodology,
    experiment_design,
    performance_data,
    with_deep_dive_plots,
    return_intermediate_data,
):
  assert experiment_design.methodology == methodology
  assign_geos(experiment_design, performance_data)
  output = analyze_experiment(
      experiment_design,
      performance_data,
      "2024-02-01",
      with_deep_dive_plots=with_deep_dive_plots,
      return_intermediate_data=return_intermediate_data,
  )
  if return_intermediate_data:
    analysis_results, intermediate_data = output
    assert isinstance(intermediate_data, dict)
  else:
    analysis_results = output

  assert isinstance(analysis_results, pd.DataFrame)

  # Check absolute values are not NA
  assert analysis_results["point_estimate"].notna().all()
  assert analysis_results["lower_bound"].notna().all()
  assert analysis_results["upper_bound"].notna().all()

  plt.close("all")
