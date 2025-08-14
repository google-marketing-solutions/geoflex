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

"""Bootstrap methods for GeoFleX."""

import logging
from typing import Any, Generator
import numpy as np
import pandas as pd
from statsmodels.tsa import seasonal


logger = logging.getLogger(__name__)


def _shuffle_columns(x: np.ndarray, n_samples: int, rng: np.random.Generator):
  """Returns a matrix where each column is a shuffled version of x.

  Args:
      x: 1D numpy array of length M.
      n_samples: Number of columns in the output matrix.
      rng: Random number generator.

  Returns:
      A (M x n_samples) numpy array where each column is a shuffled version of
      x.
  """
  n_rows = len(x)
  y = np.tile(x, (n_samples, 1)).T
  indices = rng.random((n_rows, n_samples)).argsort(axis=0)
  return y[indices, np.arange(n_samples)]


class MultivariateTimeseriesBootstrap:
  """A bootstrap method for multivariate time series.

  First separates the time series into a trend and noise component. The trend is
  differenced and then the noise and differenced trend are block bootstrapped
  to create the bootstrap samples.

  The model can be either additive:
    y_i = trend_i + noise_i
  or multiplicative:
    y_i = exp(trend_i) * noise_i

  Multiplicative is default but is only allowed for non-negative data. It
  ensures that the resulting time series is always non-negaitve, which is the
  typical case for most geo tesitng performance metrics.

  The sampling can be either permutation or random. Permutation means it
  shuffles the blocks, while random means it samples with random from the
  blocks. If permutation is used and the time series is not divisible by the
  block size, one final block is selected randomly to make sure that the final
  length of the time series is the same as the original.
  """

  # max lag to consider.
  def __init__(
      self,
      seasonality: int = 7,
      seasons_per_block: int = 4,
      model_type: str = "multiplicative",
      sampling_type: str = "permutation",
      stl_params: dict[str, Any] | None = None,
  ):
    """Initializes the MultivariateTimeseriesBootstrap.

    Args:
      seasonality: The seasonality of the time series. Defaults to 7 for daily
        data with a weekly seasonality.
      seasons_per_block: The number of complete seasons in each block. Defaults
        to 4.
      model_type: Either additive or multiplicative. Defaults to additive.
        Multiplicative is only allowed for non-negative data.
      sampling_type: Either permutation or random. Permutation means it shuffles
        the blocks, while random means it samples with random from the blocks.
        If permutation is used and the time series is not divisible by the block
        size, one final block is selected randomlyto make sure that the final
        length of the time series is the same as the original.
      stl_params: The parameters for the STL model. Defaults to 0 for seasonal
        degree, 0 for trend degree, 0 for low pass degree, and False for robust.
    """

    if model_type not in ["additive", "multiplicative"]:
      error_message = "Residual type must be either additive or multiplicative."
      logger.error(error_message)
      raise ValueError(error_message)
    if sampling_type not in ["permutation", "random"]:
      error_message = "Sampling type must be either permutation or random."
      logger.error(error_message)
      raise ValueError(error_message)

    self.model_type = model_type
    self.seasonality = seasonality
    self.seasons_per_block = seasons_per_block
    self.block_size = seasonality * seasons_per_block
    self.sampling_type = sampling_type

    if self.block_size <= 2:
      raise ValueError("Block size must be greater than 2.")

    default_stl_params = dict(
        seasonal_deg=0, trend_deg=0, low_pass_deg=0, robust=False
    )
    stl_params = stl_params or {}
    self.stl_params = default_stl_params | stl_params

  def _remove_trend(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Removes the trend from the data.

    Uses STL to remove the trend.

    Args:
      y: The data to remove the trend from. This is a (time_steps, n_series)
        numpy array where each row is a time step and each column is a series.

    Returns:
      The trend and the data with the trend removed.
    """
    trend = np.zeros_like(y)
    noise = np.zeros_like(y)

    for i, y_i in enumerate(y.T):
      stl = seasonal.STL(y_i, period=self.seasonality, **self.stl_params)
      res = stl.fit()

      if self.model_type == "additive":
        trend[:, i] = res.trend
        noise[:, i] = y_i - trend[:, i]
      else:  # self.model_type == "multiplicative"
        trend[:, i] = np.clip(res.trend, a_min=1e-6, a_max=None)
        noise[:, i] = y_i / trend[:, i]

    return trend, noise

  def fit(self, original_data: pd.DataFrame) -> None:
    """Fits the MultivariateTimeseriesBootstrap to the data.

    Args:
      original_data: The timeseries to fit. This is a dataframe where rows
        indicate the time steps and columns indicate the series.
    """
    logger.info(
        "Fitting MultivariateTimeseriesBootstrap to data, this may take a few"
        " minutes."
    )

    self._original_data = original_data.astype(float).copy()
    y = self._original_data.values.copy()

    if (self.model_type == "multiplicative") and (y < 0.0).any():
      raise ValueError(
          "Multiplicative residual type is only allowed for non-negative data."
      )

    self.n_time_steps, self.n_series = y.shape
    self.trend, self.noise = self._remove_trend(y)
    self.trend_mean = self.trend.mean(axis=0)

    if self.model_type == "multiplicative":
      self.trend = np.log(self.trend + 1e-6)  # Ensures non-negativity
    self.trend_diff = np.concatenate(
        [np.zeros((1, self.n_series)), np.diff(self.trend, axis=0)], axis=0
    )

    self._remainder_length = self.n_time_steps % self.block_size
    self.n_time_steps_complete_blocks = (
        self.n_time_steps - self._remainder_length
    )
    self.trend_diff_blocks = self.trend_diff[
        : self.n_time_steps_complete_blocks, :
    ].reshape(-1, self.block_size, self.n_series)
    self.noise_blocks = self.noise[
        : self.n_time_steps_complete_blocks, :
    ].reshape(-1, self.block_size, self.n_series)
    self.blocks_idx = np.arange(self.noise_blocks.shape[0])

    # Remainder blocks to handle the case where the time series is not divisible
    # by the block size.
    padding_length = self.block_size - self._remainder_length
    trend_diff_padded = np.concatenate(
        [self.trend_diff, np.zeros((padding_length, self.n_series))]
    )
    self.trend_diff_remainder_blocks = (
        trend_diff_padded.reshape(-1, self.block_size, self.n_series)
    )[:, : self._remainder_length, :]
    noise_padded = np.concatenate(
        [self.noise, np.zeros((padding_length, self.n_series))]
    )
    self.noise_remainder_blocks = (
        noise_padded.reshape(-1, self.block_size, self.n_series)
    )[:, : self._remainder_length, :]

    logger.info("MultivariateTimeseriesBootstrap fit complete.")

  def _sample(
      self,
      n_bootstraps: int = 1,
      rng: np.random.Generator | None = None,
  ) -> np.ndarray:
    """Returns bootstrap samples from the data.

    Args:
      n_bootstraps: The number of bootstrap samples to return.
      rng: The random number generator to use. If None, a default generator will
        be used.

    Returns:
      A (n_bootstraps, timeseries_length, n_series) numpy array containing the
      bootstrap samples.
    """
    if rng is None:
      rng = np.random.default_rng()

    if self.sampling_type == "permutation":
      sample_idx = _shuffle_columns(self.blocks_idx, n_bootstraps, rng)
    else:  # self.sampling_type == "random"
      sample_idx = rng.choice(
          self.blocks_idx,
          size=(len(self.blocks_idx), n_bootstraps),
          replace=True,
      )

    trend_diff_samp = self.trend_diff_blocks[sample_idx, :, :]
    trend_diff_samp = np.transpose(trend_diff_samp, (1, 0, 2, 3)).reshape(
        n_bootstraps, -1, self.n_series
    )
    noise_samp = self.noise_blocks[sample_idx, :, :]
    noise_samp = np.transpose(noise_samp, (1, 0, 2, 3)).reshape(
        n_bootstraps, -1, self.n_series
    )

    remainder_block_idx = np.random.choice(
        len(self.trend_diff_remainder_blocks), size=n_bootstraps, replace=True
    )
    trend_diff_remainder_block_samp = self.trend_diff_remainder_blocks[
        remainder_block_idx, :, :
    ].copy()
    remainder_block_idx = np.random.choice(
        len(self.trend_diff_remainder_blocks), size=n_bootstraps, replace=True
    )
    noise_remainder_block_samp = self.noise_remainder_blocks[
        remainder_block_idx, :, :
    ].copy()
    trend_diff_samp = np.concatenate(
        [trend_diff_samp, trend_diff_remainder_block_samp], axis=1
    )
    noise_samp = np.concatenate(
        [noise_samp, noise_remainder_block_samp], axis=1
    )

    trend_samp = trend_diff_samp.cumsum(axis=1)
    if self.model_type == "additive":
      offsets = self.trend_mean - trend_samp.mean(axis=1, keepdims=True)
      complete_samples = trend_samp + noise_samp + offsets
    else:  # self.model_type == "multiplicative"
      trend_samp = np.exp(trend_samp)
      offsets = self.trend_mean / trend_samp.mean(axis=1, keepdims=True)
      complete_samples = trend_samp * noise_samp * offsets

    return complete_samples

  def sample_dataframes(
      self,
      n_bootstraps: int = 1,
      rng: np.random.Generator | None = None,
  ) -> Generator[pd.DataFrame, None, None]:
    """Yields bootstrap samples from the data as dataframes.

    Args:
      n_bootstraps: The number of bootstrap samples to return.
      rng: The random number generator to use. If None, a default generator will
        be used.

    Yields:
      A dataframe containing the bootstrap samples.
    """
    for sample in self._sample(n_bootstraps, rng):
      yield pd.DataFrame(
          sample,
          columns=self._original_data.columns,
          index=self._original_data.index,
      )
