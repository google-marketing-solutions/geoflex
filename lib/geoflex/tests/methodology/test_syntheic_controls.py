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

"""Tests for the Synthetic Controls module."""
from typing import Any

from geoflex.data import GeoPerformanceDataset
from geoflex.experiment_design import ExperimentDesign
from geoflex.experiment_design import GeoAssignment
from geoflex.methodology.synthetic_controls import SyntheticControls
from geoflex.metrics import Metric
import pandas as pd
import pytest

# pylint:disable=protected-access
# pylint: disable=redefined-outer-name


def setup_fixtures_data() -> dict[str, Any]:
  """Sets up all common data and objects needed for the tests."""
  data = {
      'date': [
          '2025-01-01', '2025-01-01', '2025-01-01', '2025-01-01', '2025-01-01',
          '2025-01-02', '2025-01-02', '2025-01-02', '2025-01-02', '2025-01-02',
          '2025-01-03', '2025-01-03', '2025-01-03', '2025-01-03', '2025-01-03',
          '2025-01-04', '2025-01-04', '2025-01-04', '2025-01-04', '2025-01-04',
          '2025-01-05', '2025-01-05', '2025-01-05', '2025-01-05', '2025-01-05',
          '2025-01-06', '2025-01-06', '2025-01-06', '2025-01-06', '2025-01-06',
          '2025-01-07', '2025-01-07', '2025-01-07', '2025-01-07', '2025-01-07',
          '2025-01-08', '2025-01-08', '2025-01-08', '2025-01-08', '2025-01-08',
          ],
      'geo_id': ['G1', 'G2', 'G3', 'G4', 'G5'] * 8,
      'sales': [100, 200, 300, 400, 500, 110, 220, 330, 440, 550, 105, 215,
                315, 415, 515, 115, 225, 325, 425, 525, 100, 200, 300, 400,
                500, 110, 220, 330, 440, 550, 105, 215, 315, 415, 515, 115,
                225, 325, 425, 525]
    }
  sample_df = pd.DataFrame(data)

  fixtures = {
      'geo_col': 'geo_id',
      'date_col': 'date',
      'dependent_var': 'sales',
      'sc_method': SyntheticControls(),
      'historical_data': GeoPerformanceDataset(
          data=sample_df, geo_id_column='geo_id', date_column='date'
      ),
      'experiment_design': ExperimentDesign(
          primary_metric=Metric(name='sales'),
          methodology='SyntheticControls',
          runtime_weeks=4,
          n_cells=2,
          methodology_parameters={'min_treatment_geos': 2, 'num_iterations': 5},
      )
    }
  return fixtures


@pytest.fixture(name='fixtures')
def fixtures_fixture() -> dict[str, Any]:
  return setup_fixtures_data()


def setup_aggregated_df(fixtures: dict[str, Any]):
  """Sets up the aggregated dataframe."""
  return fixtures['sc_method']._aggregate_treatment(
      df=fixtures['historical_data'].parsed_data,
      treatment_geos=['G1', 'G2'],
      control_geos=['G3', 'G4', 'G5'],
      geo_var=fixtures['geo_col'],
      time_var=fixtures['date_col'],
      dependent=fixtures['dependent_var']
  )


@pytest.fixture(name='aggregated_df')
def aggregated_df_fixture(fixtures):
  return setup_aggregated_df(fixtures)


def setup_fit_results_and_validation_df(fixtures, aggregated_df):
  """Sets up the fit results and validation dataframe."""
  train_end_date = pd.to_datetime('2025-01-06')
  fit_results = fixtures['sc_method']._fit_model(
      df_agg=aggregated_df,
      geo_var=fixtures['geo_col'],
      time_var=fixtures['date_col'],
      dependent=fixtures['dependent_var'],
      control_geos=['G3', 'G4', 'G5'],
      test_geo='Aggregated_Treatment',
      treatment_geos=['G1', 'G2'],
      train_start_date=aggregated_df[fixtures['date_col']].min(),
      train_end_date=train_end_date,
      predictor_start_date=aggregated_df[fixtures['date_col']].min(),
      predictor_end_date=aggregated_df[fixtures['date_col']].max()
  )
  validation_df = aggregated_df[
      aggregated_df[fixtures['date_col']] > train_end_date]
  return fit_results, validation_df


@pytest.fixture(name='fit_results_and_validation_df')
def fit_results_and_validation_df_fixture(fixtures, aggregated_df):
  return setup_fit_results_and_validation_df(fixtures, aggregated_df)


def test_aggregate_treatment(aggregated_df):
  """Tests the treatment aggregation logic."""
  assert 'Aggregated_Treatment' in aggregated_df['geo_id'].unique()
  assert 'G1' not in aggregated_df['geo_id'].unique()


def test_fit_model(fit_results_and_validation_df):
  """Tests the core model fitting logic and returns results for other tests."""
  fit_results, _ = fit_results_and_validation_df
  assert fit_results['validation_r2'] is not None


def test_calculate_r2(fixtures: dict[str, Any], fit_results_and_validation_df):
  """Tests the R2 score calculation using a real fitted model."""
  fit_results, validation_df = fit_results_and_validation_df
  r2_score = SyntheticControls._calculate_r2(
      synth_model=fit_results['synth_model'],
      df=validation_df,
      unit_var=fixtures['geo_col'],
      time_var=fixtures['date_col'],
      dependent=fixtures['dependent_var'],
      control_geos=fit_results['control_geos'],
      test_geo='Aggregated_Treatment'
  )
  assert isinstance(r2_score, float) and r2_score <= 1.0


def test_methodology_assign_geos(fixtures: dict[str, Any]):
  """Tests the main orchestrator for the entire assignment process."""
  final_assignment, _ = fixtures['sc_method']._methodology_assign_geos(
      experiment_design=fixtures['experiment_design'],
      historical_data=fixtures['historical_data']
  )
  assert isinstance(final_assignment, GeoAssignment)

  assigned_geos = (
      set().union(*final_assignment.treatment)
      | final_assignment.control
      | final_assignment.exclude
  )
  assert assigned_geos == set(fixtures['historical_data'].geos)


def test_methodology_assign_geos_multicell(fixtures: dict[str, Any]):
  """Tests the main orchestrator for the entire assignment process."""
  fixtures['experiment_design'].n_cells = 3
  final_assignment, _ = fixtures['sc_method']._methodology_assign_geos(
      experiment_design=fixtures['experiment_design'],
      historical_data=fixtures['historical_data']
  )
  assert isinstance(final_assignment, GeoAssignment)
  assigned_geos = (
      set().union(*final_assignment.treatment)
      | final_assignment.control
      | final_assignment.exclude
  )
  assert assigned_geos == set(fixtures['historical_data'].geos)


def test_methodology_analyze_experiment(fixtures: dict[str, Any]):
  """Tests the experiment analysis logic."""

  # Use a fixed assignment for reproducibility
  geo_assignment = GeoAssignment(
      treatment=[{'G1', 'G2'}], control={'G3', 'G4', 'G5'})
  fixtures['experiment_design'].geo_assignment = geo_assignment

  # Define dates for the analysis
  pretest_end_date = pd.to_datetime('2025-01-04')
  experiment_start_date = pd.to_datetime('2025-01-05')
  experiment_end_date = pd.to_datetime('2025-01-08')

  # Run the analysis
  results_df, _ = fixtures['sc_method']._methodology_analyze_experiment(
      runtime_data=fixtures['historical_data'],
      experiment_design=fixtures['experiment_design'],
      experiment_start_date=experiment_start_date,
      experiment_end_date=experiment_end_date,
      pretest_period_end_date=pretest_end_date,
  )

  # Assertions
  assert not results_df.empty
  assert 'metric' in results_df.columns
  assert 'point_estimate' in results_df.columns
  assert 'p_value' in results_df.columns
  assert results_df['metric'].iloc[0] == 'sales'


def test_analyze_experiment_with_invalid_dates(fixtures: dict[str, Any]):
  """Tests that an error is raised with invalid date ranges."""
  geo_assignment = GeoAssignment(
      treatment=[{'G1', 'G2'}], control={'G3', 'G4', 'G5'}
  )
  fixtures['experiment_design'].geo_assignment = geo_assignment

  # Invalid dates that will cause the error
  pretest_end_date = pd.to_datetime('2025-01-01')
  experiment_start_date = pd.to_datetime('2026-01-01')
  experiment_end_date = pd.to_datetime('2026-01-08')

  with pytest.raises(
      RuntimeError,
      match='No data in the pretest or runtime period for SyntheticControls'
  ):
    fixtures['sc_method']._methodology_analyze_experiment(
        runtime_data=fixtures['historical_data'],
        experiment_design=fixtures['experiment_design'],
        experiment_start_date=experiment_start_date,
        experiment_end_date=experiment_end_date,
        pretest_period_end_date=pretest_end_date,
    )


if __name__ == '__main__':
  fixtures_data = setup_fixtures_data()
  aggregated_data = setup_aggregated_df(fixtures_data)

  test_aggregate_treatment(aggregated_data)

  fit_results, validation_df = setup_fit_results_and_validation_df(
      fixtures_data, aggregated_data
  )
  test_fit_model((fit_results, validation_df))
  test_calculate_r2(fixtures_data, (fit_results, validation_df))

  test_methodology_assign_geos(fixtures_data)
  test_methodology_assign_geos_multicell(fixtures_data)
  test_methodology_analyze_experiment(fixtures_data)
