"""Bootstrap methods for GeoFleX."""

import logging
from typing import Generator
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.seasonal import DecomposeResult
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.vector_ar.var_model import VAR

logger = logging.getLogger(__name__)


def _auto_seasonal_decompose(
    y: np.ndarray,
    period: int = 7,
    seasons_per_filt: int | None = None,
    max_seasons_per_filt: int = 4,
) -> DecomposeResult:
  """Automatically selects the best seasonal decomposition for the data.

  It does this by selecting the best trend filtering so that the noise component
  is as close as possible to AR(1).

  Args:
      y: The timeseries to decompose.
      period: The period of the seasonality.
      seasons_per_filt: The number of seasons per filter. If None, the best
        filtering is automatically selected.
      max_seasons_per_filt: The maximum number of seasons per filter to
        consider.

  Returns:
      The best seasonal decomposition.
  """
  if seasons_per_filt is not None:
    filt = np.ones(seasons_per_filt * period) / float(seasons_per_filt * period)
    return seasonal_decompose(y, period=period, filt=filt, extrapolate_trend=3)
  best_aic = np.inf
  best_decomp = None
  for i in range(1, max_seasons_per_filt + 1):
    filt = np.ones(i * period) / float(i * period)
    decomp = seasonal_decompose(
        y, period=period, filt=filt, extrapolate_trend=3
    )
    ar1_fit = VAR(decomp.resid).fit(1)
    if ar1_fit.aic < best_aic:
      best_aic = ar1_fit.aic
      best_decomp = decomp
  return best_decomp


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
  """An implementation of bootstrapping for multivariate time series data.

  The algorithm is an adaptation of simple block bootstrapping: 1. The data is
  seasonal decomposed into trend, seasonal and noise components. 2. The trend is
  differenced to make it stationary. 3. The noise and differenced trend are
  split into blocks of size block_size. 4. For each bootstrap sample, the blocks
  are shuffled. 5. Finally the bootstrapped noise and trend are added to the
  seasonal

     component to get the bootstrapped time series.
  Optional settings:
    log_transform: If True, the data is log transformed before bootstrapping.
    seasons_per_block: The number of seasons per block. Can be used to override
      the automatically inferred block size in the algorithm above.
    verbose: If True, prints some diagnostic information.
    full_blocks_only: If True, only full blocks are used for bootstrapping. This
      means that the last few data points (after the final complete block) are
      ignored. Runs faster, but the resulting bootstrap samples are not the same
      size as the original data.
  """

  # max lag to consider.
  def __init__(
      self,
      seasonality: int = 7,
      log_transform: bool = False,
      seasons_per_block: int | None = None,
      verbose: bool = False,
      full_blocks_only: bool = False,
      max_lag: int = 30,
  ):
    """Initializes the MultivariateTimeseriesBootstrap.

    Args:
      seasonality: The seasonality of the data.
      log_transform: If True, the data is log transformed before bootstrapping.
        Useful to ensure that the data is positive.
      seasons_per_block: The number of seasons per block. Can be used to
        override the automatically inferred block size in the algorithm above.
      verbose: If True, prints some diagnostic information.
      full_blocks_only: If True, only full blocks are used for bootstrapping.
        This means that the last few data points (after the final complete
        block) are ignored. Runs faster, but the resulting bootstrap samples are
        not the same size as the original data.
      max_lag: The maximum lag to consider when fitting the AR model. This is
        used to determine the block size.
    """
    self.seasonality = seasonality
    self.log_transform = log_transform
    self.seasons_per_block = seasons_per_block
    self.verbose = verbose
    self.full_blocks_only = full_blocks_only
    self.max_lag = max_lag
    # Set when fit() is called.
    self.block_size = None

  def _transform_y(self, y: np.ndarray) -> np.ndarray:
    """Transforms the data before bootstrapping."""
    if self.log_transform:
      return np.log1p(y)
    else:
      return y

  def _inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
    """Inverses the transformation of the data before output."""
    if self.log_transform:
      return np.expm1(y)
    else:
      return y

  def _set_block_size(self, y_diff: np.ndarray) -> np.ndarray:
    """Sets the block size based on the autocorrelation of the data."""
    max_order = 1
    max_lag = max([min([self.max_lag, y_diff.shape[0] // 2 - 1]), 1])

    for i in range(self.n_series):
      ar_lags = ar_select_order(y_diff[:, i], maxlag=max_lag).ar_lags
      if ar_lags is not None:
        max_order = int(np.max(np.concatenate([ar_lags, [max_order]])))
    min_seasons_per_block = int(np.ceil(max_order / self.seasonality))
    if self.seasons_per_block is None:
      # Automatically infer the block size as 1 higher than the minimum
      self.seasons_per_block = min_seasons_per_block + 1
      if self.verbose:
        logger.info(
            "Automatically inferred block size = %s (%s seasons)",
            self.seasons_per_block * self.seasonality,
            self.seasons_per_block,
        )
    elif self.seasons_per_block < min_seasons_per_block:
      logger.warning(
          "The seasons_per_block=%s is too small, it does not capture the "
          "autocorrelation in the data. Try a bigger one.",
          self.seasons_per_block,
      )
    self.block_size = self.seasons_per_block * self.seasonality

  def fit(
      self, y: np.ndarray | pd.DataFrame, seasons_per_filt: int | None = 2
  ) -> None:
    """Fits the MultivariateTimeseriesBootstrap to the data.

    Args:
      y: The timeseries to fit. The shape is (n_time_steps, n_series).
      seasons_per_filt: The number of seasons per filter window to use when
        decomposing the data into trend, seasonal and noise components. If None,
        the best filtering is automatically selected.
    """
    if isinstance(y, pd.DataFrame):
      self._columns = y.columns.copy()
      self._index = y.index.copy()
      y = y.values
    else:
      self._columns = [f"Column {i}" for i in range(y.shape[1])]
      self._index = np.arange(y.shape[0])

    # Check that the number of time steps is at least 4 * seasonality.
    if self.seasonality * seasons_per_filt * 2 > y.shape[0]:
      raise ValueError(
          "The number of time steps is too small for the given seasonality. "
          "Please increase the number of time steps or decrease the "
          "seasonality."
      )

    self.n_series = y.shape[1]
    self.n_time_steps = y.shape[0]
    self.y_transformed = self._transform_y(y).copy()
    decomp = _auto_seasonal_decompose(
        self.y_transformed,
        period=self.seasonality,
        seasons_per_filt=seasons_per_filt,
    )

    self.trend = decomp.trend
    self.seasonal = decomp.seasonal
    self.noise = decomp.resid
    self.trend_diff = np.concatenate(
        [np.zeros((1, self.n_series)), diff(self.trend, k_diff=1)], axis=0
    )
    self._set_block_size(self.trend_diff)
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
    if not self.full_blocks_only:
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

  def sample(
      self,
      n_bootstraps: int = 1,
      index_point: int | None = None,
      rng: np.random.Generator | None = None,
  ) -> np.ndarray:
    """Returns bootstrap samples from the data.

    Args:
      n_bootstraps: The number of bootstrap samples to return.
      index_point: The index point to align the bootstrap samples to. If None,
        the means are aligned.
      rng: The random number generator to use. If None, a default generator will
        be used.

    Returns:
      A (n_bootstraps, timeseries_length, n_series) numpy array containing the
      bootstrap samples.
    """
    if rng is None:
      rng = np.random.default_rng(0)
    sample_idx = _shuffle_columns(self.blocks_idx, n_bootstraps, rng)
    trend_diff_samp = self.trend_diff_blocks[sample_idx, :, :]
    trend_diff_samp = np.transpose(trend_diff_samp, (1, 0, 2, 3)).reshape(
        n_bootstraps, -1, self.n_series
    )
    sample_idx = _shuffle_columns(self.blocks_idx, n_bootstraps, rng)
    noise_samp = self.noise_blocks[sample_idx, :, :]
    noise_samp = np.transpose(noise_samp, (1, 0, 2, 3)).reshape(
        n_bootstraps, -1, self.n_series
    )
    if not self.full_blocks_only:
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
    y_transform_samp = (
        noise_samp
        + trend_diff_samp.cumsum(axis=1)
        + self.remove_remainder(self.seasonal)[np.newaxis, :, :]
    )
    if index_point is not None:
      # Index so that the values are equal at the index point
      index_points = rng.integers(
          self.n_time_steps_complete_blocks, size=n_bootstraps
      )
      initial_values = self.y_transformed[index_points, np.newaxis, :]
      y_transform_samp += (
          initial_values
          - y_transform_samp[
              np.arange(n_bootstraps), index_points, np.newaxis, :
          ]
      )
    else:
      # Offset so that the means are equal
      y_transform_samp += self.y_transformed.mean(
          axis=0
      ) - y_transform_samp.mean(axis=1, keepdims=True)
    return self._inverse_transform_y(y_transform_samp)

  def sample_dataframes(
      self,
      n_bootstraps: int = 1,
      index_point: int | None = None,
      rng: np.random.Generator | None = None,
  ) -> Generator[pd.DataFrame, None, None]:
    """Yields bootstrap samples from the data as dataframes."""
    for sample in self.sample(n_bootstraps, index_point, rng):
      yield pd.DataFrame(sample, columns=self._columns, index=self._index)

  def remove_remainder(self, y: np.ndarray) -> np.ndarray:
    """If using full_blocks_only this removes the end of the timeseries.

    Useful for aligning the original data with the bootstrap samples if the
    original data is not a multiple of the block size.

    Args:
      y: The original data.

    Returns:
      The original data with the end of the timeseries removed.
    """
    if self.full_blocks_only:
      return y[: self.n_time_steps_complete_blocks, :].copy()
    else:
      return y
