"""Utils for GeoFleX."""

import base64
import io
from typing import Annotated
import pandas as pd
import pydantic


def serialize_dataframe_to_parquet_b64(data: pd.DataFrame) -> str:
  """Serializes a DataFrame to a Base64 encoded Parquet string.

  Used with pydantic for serialization of DataFrames.

  Args:
    data: The DataFrame to serialize.

  Returns:
    The Base64 encoded Parquet string.
  """
  # if data.empty and not data.columns.tolist():
  #   pass
  parquet_bytes = data.to_parquet()
  return base64.b64encode(parquet_bytes).decode('utf-8')


def deserialize_df_from_parquet_b64_if_string(value: str) -> pd.DataFrame:
  """Deserializes a DataFrame from a Base64 encoded Parquet string.

  Used with pydantic for deserialization of DataFrames from json.

  Args:
    value: The Base64 encoded Parquet string.

  Returns:
    The deserialized DataFrame.
  """
  if not isinstance(value, str):  # Not a string
    return value

  if not value:  # Empty string
    return pd.DataFrame()

  try:
    base64_decoded_bytes = base64.b64decode(value.encode('utf-8'))
    if not base64_decoded_bytes:  # Empty bytes after decoding
      return pd.DataFrame()
    parquet_file_like = io.BytesIO(base64_decoded_bytes)
    return pd.read_parquet(parquet_file_like)
  except Exception as e:
    raise ValueError(
        f'Failed to deserialize DataFrame from Parquet string: {e}'
    ) from e


# Use this type annotation instead of pd.DataFrame to serialize and deserialize
# DataFrames to and from Parquet strings.
ParquetDataFrame = Annotated[
    pd.DataFrame,
    pydantic.PlainSerializer(
        serialize_dataframe_to_parquet_b64, when_used='json'
    ),
    pydantic.BeforeValidator(deserialize_df_from_parquet_b64_if_string),
]
