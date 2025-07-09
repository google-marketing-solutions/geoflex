"""Tests for the visualization module."""

import geoflex.data
import geoflex.experiment_design
import geoflex.methodology
import geoflex.visualization
import numpy as np
import pandas as pd
import pytest


# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


@pytest.fixture(name="performance_data")
def performance_data_fixture():
  """Fixture for a mock historical data with lots of geos."""
  rng = np.random.default_rng(seed=42)
  data = pd.DataFrame({
      "geo_id": [f"geo_{i}" for i in range(20) for _ in range(100)],  # pylint: disable=g-complex-comprehension
      "date": pd.date_range(start="2024-01-01", periods=100).tolist() * 20,
      "revenue": rng.random(size=2000),
      "cost": rng.random(size=2000),
      "conversions": rng.random(size=2000),
  })
  data["date"] = data["date"].dt.strftime("%Y-%m-%d")

  return geoflex.data.GeoPerformanceDataset(data=data)


@pytest.fixture(name="experiment_design")
def experiment_design_fixture():
  """Fixture for a mock experiment design."""
  return geoflex.experiment_design.ExperimentDesign(
      primary_metric="revenue",
      secondary_metrics=["conversions"],
      methodology="TestingMethodology",
      runtime_weeks=2,
      n_cells=2,
      alpha=0.1,
  )


def test_display_analysis_results_works(experiment_design, performance_data):
  # First run experiment to get results
  geoflex.methodology.assign_geos(experiment_design, performance_data)
  analysis_results = geoflex.methodology.analyze_experiment(
      experiment_design, performance_data, "2024-02-01"
  )

  # Test that the visualization works
  visualization = geoflex.visualization.display_analysis_results(
      analysis_results, 0.1
  )
  assert isinstance(visualization, pd.io.formats.style.Styler)
  assert visualization.to_html()  # Check that the html can be generated
