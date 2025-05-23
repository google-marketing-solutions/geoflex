"""Tests for the utils module."""

import re
from feedx import statistics
from geoflex import utils
import numpy as np
import pandas as pd
import pydantic
import pytest


ParquetDataFrame = utils.ParquetDataFrame
assign_geos_randomly = utils.assign_geos_randomly

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=g-explicit-bool-comparison


@pytest.fixture(name="TestParquetDataFrame")
def mock_pydantic_model_fixture():
  """Fixture for a mock pydantic model."""

  class TestParquetDataFrame(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    a: int
    required_df: ParquetDataFrame
    optional_df: ParquetDataFrame | None = None
    list_of_dfs: list[ParquetDataFrame] = []

  return TestParquetDataFrame


@pytest.fixture(name="mock_dataframe")
def mock_dataframe_fixture():
  """Fixture for a mock dataframe."""
  return pd.DataFrame(
      {"a": [1, 2, None], "b": ["4", None, "6"], "c": [[7, 8], None, [10]]}
  )


@pytest.fixture(name="mock_empty_dataframe")
def mock_empty_dataframe_fixture():
  """Fixture for a mock empty dataframe."""
  return pd.DataFrame()


@pytest.fixture(name="mock_empty_dataframe_with_columns")
def mock_empty_dataframe_with_columns_fixture():
  """Fixture for a mock empty dataframe with columns."""
  return pd.DataFrame(columns=["a", "b", "c"])


def test_serialize_deserialize_parquet_dataframe_optional_unset(
    TestParquetDataFrame, mock_dataframe
):
  model = TestParquetDataFrame(
      a=1,
      required_df=mock_dataframe,
  )

  json_string = model.model_dump_json()
  new_model = TestParquetDataFrame.model_validate_json(json_string)

  pd.testing.assert_frame_equal(
      new_model.required_df, mock_dataframe, check_like=True
  )
  assert new_model.optional_df is None
  assert new_model.list_of_dfs == []
  assert new_model.a == 1


def test_serialize_deserialize_parquet_dataframe_optional_set(
    TestParquetDataFrame, mock_dataframe
):
  model = TestParquetDataFrame(
      a=1,
      required_df=mock_dataframe,
      optional_df=mock_dataframe.copy(),
  )

  json_string = model.model_dump_json()
  new_model = TestParquetDataFrame.model_validate_json(json_string)

  pd.testing.assert_frame_equal(
      new_model.required_df, mock_dataframe, check_like=True
  )
  pd.testing.assert_frame_equal(
      new_model.optional_df, mock_dataframe, check_like=True
  )
  assert new_model.list_of_dfs == []
  assert new_model.a == 1


def test_serialize_deserialize_parquet_dataframe_list_of_dfs(
    TestParquetDataFrame, mock_dataframe
):
  model = TestParquetDataFrame(
      a=1,
      required_df=mock_dataframe,
      list_of_dfs=[mock_dataframe.copy(), mock_dataframe.copy()],
  )

  json_string = model.model_dump_json()
  new_model = TestParquetDataFrame.model_validate_json(json_string)

  pd.testing.assert_frame_equal(
      new_model.required_df, mock_dataframe, check_like=True
  )
  assert len(new_model.list_of_dfs) == 2
  pd.testing.assert_frame_equal(
      new_model.list_of_dfs[0], mock_dataframe, check_like=True
  )
  pd.testing.assert_frame_equal(
      new_model.list_of_dfs[1], mock_dataframe, check_like=True
  )
  assert new_model.optional_df is None
  assert new_model.a == 1


def test_serialize_deserialize_parquet_dataframe_empty_dataframe(
    TestParquetDataFrame, mock_empty_dataframe
):
  model = TestParquetDataFrame(
      a=1,
      required_df=mock_empty_dataframe,
  )

  json_string = model.model_dump_json()
  new_model = TestParquetDataFrame.model_validate_json(json_string)

  pd.testing.assert_frame_equal(
      new_model.required_df, mock_empty_dataframe, check_like=True
  )
  assert new_model.optional_df is None
  assert new_model.list_of_dfs == []
  assert new_model.a == 1


def test_serialize_deserialize_parquet_dataframe_empty_dataframe_with_columns(
    TestParquetDataFrame, mock_empty_dataframe_with_columns
):
  model = TestParquetDataFrame(
      a=1,
      required_df=mock_empty_dataframe_with_columns,
  )

  json_string = model.model_dump_json()
  new_model = TestParquetDataFrame.model_validate_json(json_string)

  pd.testing.assert_frame_equal(
      new_model.required_df, mock_empty_dataframe_with_columns, check_like=True
  )
  assert new_model.optional_df is None
  assert new_model.list_of_dfs == []
  assert new_model.a == 1


@pytest.mark.parametrize(
    "alternative_hypothesis", ["two-sided", "greater", "less"]
)
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_infer_p_value_returns_correct_p_value(alternative_hypothesis, alpha):
  rng = np.random.default_rng(seed=0)
  values1 = rng.normal(size=10000)
  values2 = rng.normal(size=10000)

  statistical_test_results = statistics.yuens_t_test_ind(
      values1=values1,
      values2=values2,
      trimming_quantile=0.0,
      alternative=alternative_hypothesis,
      alpha=alpha,
  )

  mean = statistical_test_results.absolute_difference
  confidence_interval = (
      statistical_test_results.absolute_difference_lower_bound,
      statistical_test_results.absolute_difference_upper_bound,
  )
  inferred_p_value = utils.infer_p_value(
      mean=mean,
      confidence_interval=confidence_interval,
      alpha=alpha,
      alternative_hypothesis=alternative_hypothesis,
  )

  # Only close, not identical, because the inferred p-value assumes a normal
  # distribution, while the statistical test uses a t-distribution.
  assert np.isclose(
      inferred_p_value, statistical_test_results.p_value, atol=1e-6
  )


@pytest.fixture(name="rng_fixed_seed")
def rng_fixed_seed_fixture():
  """Provides a numpy random number generator with a fixed seed."""
  return np.random.default_rng(seed=42)


def test_basic_assignment_no_constraints(rng_fixed_seed):
  """Test basic assignment with no metrics or pre-assignments."""
  geo_ids = ["G1", "G2", "G3", "G4"]
  n_groups = 2

  assigned_geos, group_metrics = assign_geos_randomly(
      geo_ids, n_groups, rng_fixed_seed
  )

  assert len(assigned_geos) == n_groups
  assert len(group_metrics) == n_groups

  all_assigned = sorted([geo for group in assigned_geos for geo in group])  # pylint: disable=g-complex-comprehension
  assert all_assigned == sorted(geo_ids)  # All geos assigned

  # Check that default metric (1.0 per geo) is applied
  for i in range(n_groups):
    assert group_metrics[i] == float(len(assigned_geos[i]))
    assert assigned_geos[i]  # No empty groups due to final check


def test_assignment_with_metrics_and_constraints(rng_fixed_seed):
  """Test assignment with specific metrics and max_metric_per_group."""
  geo_ids = ["G1", "G2", "G3", "G4", "G5"]
  metric_values = [10.0, 20.0, 5.0, 15.0, 25.0]
  n_groups = 2
  max_metric_per_group = [30.0, 40.0]

  assigned_geos, group_metrics = assign_geos_randomly(
      geo_ids, n_groups, rng_fixed_seed, metric_values, max_metric_per_group
  )

  assert len(assigned_geos) == n_groups
  assert len(group_metrics) == n_groups

  geo_to_metric_map = {gid: met for gid, met in zip(geo_ids, metric_values)}

  total_assigned_metric = 0
  all_assigned_geos_flat = []
  for i, current_assigned_geos in enumerate(assigned_geos):
    current_metric_sum = sum(geo_to_metric_map[g] for g in assigned_geos[i])
    assert abs(group_metrics[i] - current_metric_sum) < 1e-9
    assert group_metrics[i] <= max_metric_per_group[i] + 1e-9
    assert current_assigned_geos  # No empty groups
    all_assigned_geos_flat.extend(current_assigned_geos)
    total_assigned_metric += current_metric_sum


def test_pre_assigned_geos_respected(rng_fixed_seed):
  """Test that pre-assigned geos are placed in their designated groups."""
  geo_ids = ["G1", "G2", "G3", "G4"]
  n_groups = 2
  pre_assigned_geos = {"G1": 0, "G4": 1}
  metric_values = [1.0, 2.0, 3.0, 4.0]

  assigned_geos, group_metrics = assign_geos_randomly(
      geo_ids,
      n_groups,
      rng_fixed_seed,
      metric_values=metric_values,
      pre_assigned_geos=pre_assigned_geos,
  )

  assert "G1" in assigned_geos[0]
  assert "G4" in assigned_geos[1]
  assert group_metrics[0] >= 1.0  # G1's metric
  assert group_metrics[1] >= 4.0  # G4's metric

  # Ensure all geos are assigned and groups are not empty (due to final check)
  all_assigned_count = sum(len(g) for g in assigned_geos)
  assert all_assigned_count == len(geo_ids)
  assert assigned_geos[0]
  assert assigned_geos[1]


def test_all_geos_pre_assigned(rng_fixed_seed):
  """Test behavior when all geos are pre-assigned."""
  geo_ids = ["G1", "G2"]
  n_groups = 2
  metric_values = [10.0, 20.0]
  pre_assigned_geos = {"G1": 0, "G2": 1}

  assigned_geos, group_metrics = assign_geos_randomly(
      geo_ids,
      n_groups,
      rng_fixed_seed,
      metric_values,
      pre_assigned_geos=pre_assigned_geos,
  )

  assert assigned_geos == [["G1"], ["G2"]]
  assert group_metrics == [10.0, 20.0]


def test_no_geos_to_assign_empty_groups_allowed(rng_fixed_seed):
  """Test with no geos; groups should be empty, and no 'empty group' error."""
  geo_ids = []
  n_groups = 2
  metric_values = []
  max_metric_per_group = [10.0, 10.0]

  assigned_geos, group_metrics = assign_geos_randomly(
      geo_ids, n_groups, rng_fixed_seed, metric_values, max_metric_per_group
  )

  assert assigned_geos == [[], []]
  assert group_metrics == [0.0, 0.0]


@pytest.mark.parametrize("n_groups", [-1, 0])
def test_n_groups_less_than_1_error(rng_fixed_seed, n_groups):
  """Test ValueError for negative n_groups."""
  with pytest.raises(
      ValueError, match=re.escape("n_groups must be greater than 0.")
  ):
    assign_geos_randomly(
        geo_ids=["G1", "G2", "G3"], n_groups=n_groups, rng=rng_fixed_seed
    )


def test_mismatched_geo_ids_metric_values_error(rng_fixed_seed):
  """Test ValueError for mismatched lengths of geo_ids and metric_values."""
  with pytest.raises(
      ValueError,
      match=re.escape("geo_ids and metric must have the same length."),
  ):
    assign_geos_randomly(
        geo_ids=["G1", "G2"],
        n_groups=1,
        rng=rng_fixed_seed,
        metric_values=[1.0],
    )


def test_mismatched_n_groups_max_metric_error(rng_fixed_seed):
  """Test ValueError for n_groups not matching length of max_metric_per_group."""
  with pytest.raises(
      ValueError,
      match=re.escape(
          "n_groups must be equal to the length of max_metric_per_group."
      ),
  ):
    assign_geos_randomly(
        geo_ids=["G1"],
        n_groups=2,
        rng=rng_fixed_seed,
        max_metric_per_group=[10.0],
    )


def test_pre_assigned_geo_not_in_geo_ids_error(rng_fixed_seed):
  """Test ValueError if a pre-assigned geo is not in the main geo_ids list."""
  with pytest.raises(
      ValueError,
      match=re.escape(
          "Pre-assigned geo_id 'G2' not found in main geo_ids list."
      ),
  ):
    assign_geos_randomly(
        geo_ids=["G1"],
        n_groups=1,
        rng=rng_fixed_seed,
        pre_assigned_geos={"G2": 0},
    )


@pytest.mark.parametrize("pre_assignment", [-1, 2])
def test_pre_assigned_invalid_group_index_error(rng_fixed_seed, pre_assignment):
  """Test ValueError for invalid target_group_idx in pre-assignments."""
  with pytest.raises(
      ValueError,
      match=re.escape(
          f"Invalid target_group_idx {pre_assignment} for pre-assigned geo"
          " 'G1'. Must be between 0 and 1."
      ),
  ):
    assign_geos_randomly(
        geo_ids=["G1"],
        n_groups=2,
        rng=rng_fixed_seed,
        pre_assigned_geos={"G1": pre_assignment},
    )


def test_some_geos_unassigned_due_to_capacity(rng_fixed_seed):
  """Test that geos are not assigned if they exceed group capacity."""
  geo_ids = ["G1", "G2", "G3"]
  metric_values = [10.0, 10.0, 10.0]
  n_groups = 1
  max_metric_per_group = [15.0]  # Only one geo can fit

  assigned_geos, group_metrics = assign_geos_randomly(
      geo_ids, n_groups, rng_fixed_seed, metric_values, max_metric_per_group
  )

  assert len(assigned_geos[0]) == 1  # Only one geo should be assigned
  assert group_metrics[0] == 10.0


def test_empty_group_error_if_random_assignment_leaves_group_empty(
    rng_fixed_seed,
):
  """Test ValueError if random assignment results in an empty group."""
  geo_ids = ["G1"]  # Only one geo
  metric_values = [1.0]
  n_groups = 2  # Two groups, but only one geo
  max_metric_per_group = [10.0, 10.0]

  # G1 will be assigned to one group. The other will be empty.
  # This will trigger the "empty group" error because random assignment phase
  # was entered.
  with pytest.raises(
      ValueError, match=re.escape("is empty after assigning all geos")
  ):
    assign_geos_randomly(
        geo_ids, n_groups, rng_fixed_seed, metric_values, max_metric_per_group
    )


def test_pre_assignment_exceeds_capacity_no_error_at_pre_assignment(
    rng_fixed_seed,
):
  geo_ids = ["G1", "G2"]
  metric_values = [100.0, 1.0]
  n_groups = 2
  max_metric_per_group = [10.0, 10.0]  # Group 0 capacity is 10
  pre_assigned_geos = {"G1": 0}  # Pre-assign G1 (metric 100) to group 0

  assigned_geos, group_metrics = assign_geos_randomly(
      geo_ids,
      n_groups,
      rng_fixed_seed,
      metric_values,
      max_metric_per_group,
      pre_assigned_geos,
  )

  assert "G1" in assigned_geos[0]
  assert group_metrics[0] == 100.0  # Exceeds max_metric_per_group[0]
  assert "G2" in assigned_geos[1]
  assert group_metrics[1] == 1.0
  # No error raised, demonstrating current behavior.


def test_reproducibility_with_seed():
  """Test that the same seed yields the same results."""
  geo_ids = [f"g{i}" for i in range(20)]
  n_groups = 2

  rng1 = np.random.default_rng(seed=123)
  assigned1, metrics1 = assign_geos_randomly(geo_ids, n_groups, rng1)

  rng2 = np.random.default_rng(seed=123)
  assigned2, metrics2 = assign_geos_randomly(geo_ids, n_groups, rng2)

  assert assigned1 == assigned2
  assert metrics1 == metrics2

  rng3 = np.random.default_rng(seed=456)
  assigned3, _ = assign_geos_randomly(geo_ids, n_groups, rng3)
  assert assigned1 != assigned3


@pytest.mark.parametrize(
    "test_case",
    [
        # Case 1: Basic two-sided test
        dict(
            impact_estimate=10.0,
            impact_standard_error=2.0,
            degrees_of_freedom=100,
            alternative_hypothesis="two-sided",
            alpha=0.05,
            baseline_estimate=100.0,
            baseline_standard_error=5.0,
            impact_baseline_corr=0.1,
            invert_result=False,
            expected_point_estimate=10.0,
            expected_lower_bound=6.032056963100733,
            expected_upper_bound=13.967943036899267,
            expected_p_value_approx=1.58e-06,
            expected_point_estimate_relative=0.1,
            expected_lower_bound_relative=-0.0028308347773883247,
            expected_upper_bound_relative=0.22389971794504993,
        ),
        # Case 2: invert_result = True
        dict(
            impact_estimate=10.0,
            impact_standard_error=2.0,
            degrees_of_freedom=100,
            alternative_hypothesis="two-sided",
            alpha=0.05,
            baseline_estimate=100.0,
            baseline_standard_error=5.0,
            impact_baseline_corr=0.1,
            invert_result=True,
            expected_point_estimate=0.1,
            expected_lower_bound=0.0715925027298786,
            expected_upper_bound=0.16578092781901677,
            expected_p_value_approx=1.58e-06,
            expected_point_estimate_relative=pd.NA,
            expected_lower_bound_relative=pd.NA,
            expected_upper_bound_relative=pd.NA,
        ),
        # Case 3: alternative_hypothesis = "greater"
        dict(
            impact_estimate=10.0,
            impact_standard_error=2.0,
            degrees_of_freedom=100,
            alternative_hypothesis="greater",
            alpha=0.05,
            baseline_estimate=100.0,
            baseline_standard_error=5.0,
            impact_baseline_corr=0.1,
            invert_result=False,
            expected_point_estimate=10.0,
            expected_lower_bound=6.6795313478685,
            expected_upper_bound=np.inf,
            expected_p_value_approx=1.225086706751899e-06,
            expected_point_estimate_relative=0.1,
            expected_lower_bound_relative=0.01275282756255125,
            expected_upper_bound_relative=np.inf,
        ),
        # Case 4: alternative_hypothesis = "less"
        dict(
            impact_estimate=-5.0,
            impact_standard_error=1.0,
            degrees_of_freedom=100,
            alternative_hypothesis="less",
            alpha=0.05,
            baseline_estimate=50.0,
            baseline_standard_error=2.0,
            impact_baseline_corr=0.2,
            invert_result=False,
            expected_point_estimate=-5.0,
            expected_lower_bound=-np.inf,
            expected_upper_bound=-3.33976567393425,
            expected_p_value_approx=7.9e-07,
            expected_point_estimate_relative=-0.1,
            expected_lower_bound_relative=-1.0,
            expected_upper_bound_relative=-0.033920373939951176,
        ),
        # Case 5: baseline_estimate is None
        dict(
            impact_estimate=10.0,
            impact_standard_error=2.0,
            degrees_of_freedom=100,
            alternative_hypothesis="two-sided",
            alpha=0.05,
            baseline_estimate=None,
            baseline_standard_error=None,
            impact_baseline_corr=None,
            invert_result=False,
            expected_point_estimate=10.0,
            expected_lower_bound=6.032056963100733,
            expected_upper_bound=13.967943036899267,
            expected_p_value_approx=1.58e-06,
            expected_point_estimate_relative=pd.NA,
            expected_lower_bound_relative=pd.NA,
            expected_upper_bound_relative=pd.NA,
        ),
        # Case 6: baseline_estimate is not positive
        dict(
            impact_estimate=10.0,
            impact_standard_error=2.0,
            degrees_of_freedom=100,
            alternative_hypothesis="two-sided",
            alpha=0.05,
            baseline_estimate=-100.0,
            baseline_standard_error=5.0,
            impact_baseline_corr=0.1,
            invert_result=False,
            expected_point_estimate=10.0,
            expected_lower_bound=6.032056963100733,
            expected_upper_bound=13.967943036899267,
            expected_p_value_approx=1.58e-06,
            expected_point_estimate_relative=pd.NA,
            expected_lower_bound_relative=pd.NA,
            expected_upper_bound_relative=pd.NA,
        ),
        # Case 7: baseline_estimate + impact_estimate is not positive
        dict(
            impact_estimate=-110.0,
            impact_standard_error=2.0,
            degrees_of_freedom=100,
            alternative_hypothesis="two-sided",
            alpha=0.05,
            baseline_estimate=100.0,
            baseline_standard_error=5.0,
            impact_baseline_corr=0.1,
            invert_result=False,
            expected_point_estimate=-110.0,
            expected_lower_bound=-113.96794303689927,
            expected_upper_bound=-106.03205696310073,
            expected_p_value_approx=0.0,
            expected_point_estimate_relative=pd.NA,
            expected_lower_bound_relative=pd.NA,
            expected_upper_bound_relative=pd.NA,
        ),
        # Case 8: invert_result = True, negative impact
        dict(
            impact_estimate=-10.0,
            impact_standard_error=2.0,
            degrees_of_freedom=100,
            alternative_hypothesis="two-sided",
            alpha=0.05,
            baseline_estimate=100.0,
            baseline_standard_error=5.0,
            impact_baseline_corr=0.1,
            invert_result=True,
            expected_point_estimate=-0.1,
            expected_lower_bound=-0.07159250272,
            expected_upper_bound=-0.16578092,
            expected_p_value_approx=1.58e-06,
            expected_point_estimate_relative=pd.NA,
            expected_lower_bound_relative=pd.NA,
            expected_upper_bound_relative=pd.NA,
        ),
    ],
)
def test_get_summary_statistics_from_standard_errors(test_case):
  """Tests get_summary_statistics_from_standard_errors."""
  results = utils.get_summary_statistics_from_standard_errors(
      impact_estimate=test_case["impact_estimate"],
      impact_standard_error=test_case["impact_standard_error"],
      degrees_of_freedom=test_case["degrees_of_freedom"],
      alternative_hypothesis=test_case["alternative_hypothesis"],
      alpha=test_case["alpha"],
      baseline_estimate=test_case["baseline_estimate"],
      baseline_standard_error=test_case["baseline_standard_error"],
      impact_baseline_corr=test_case["impact_baseline_corr"],
      invert_result=test_case["invert_result"],
  )

  assert results["point_estimate"] == pytest.approx(
      test_case["expected_point_estimate"]
  )
  assert results["lower_bound"] == pytest.approx(
      test_case["expected_lower_bound"], nan_ok=True
  )
  assert results["upper_bound"] == pytest.approx(
      test_case["expected_upper_bound"], nan_ok=True
  )
  assert results["p_value"] == pytest.approx(
      test_case["expected_p_value_approx"], abs=1e-5
  )

  if pd.isna(test_case["expected_point_estimate_relative"]):
    assert pd.isna(results["point_estimate_relative"])
  else:
    assert results["point_estimate_relative"] == pytest.approx(
        test_case["expected_point_estimate_relative"]
    )

  if pd.isna(test_case["expected_lower_bound_relative"]):
    assert pd.isna(results["lower_bound_relative"])
  else:
    assert results["lower_bound_relative"] == pytest.approx(
        test_case["expected_lower_bound_relative"], nan_ok=True
    )

  if pd.isna(test_case["expected_upper_bound_relative"]):
    assert pd.isna(results["upper_bound_relative"])
  else:
    assert results["upper_bound_relative"] == pytest.approx(
        test_case["expected_upper_bound_relative"], nan_ok=True
    )
