"""Tests for the utils module."""

from geoflex import utils
import pandas as pd
import pydantic
import pytest

ParquetDataFrame = utils.ParquetDataFrame

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
