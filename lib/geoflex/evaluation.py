"""Methods to evaluate the quality of a geo experiments."""

import functools
import dcor
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score


class GeoAssignmentRepresentivenessScorer:
  """Scores the representativeness of a geo assignment.

  Ideally, each of the treatment groups in a geo assignment should be
  representative ofthe entire population. This scorer uses the silhouette score,
  in combination with distance correlation, to measure the representativeness of
  the treatment groups. 1. The distance correlation is used to measure the
  "distance" between two

    geos. It is defined as `1 - the distance correlation coefficient`. This
    gives a distance matrix for all the geos. The distance correlation
    coefficient is a multivariate correlation which evaluates the similarity
    based on all numerical columns in the data. For more information see:
    https://arxiv.org/pdf/0803.4101
  2. The silhouette score is used to measure the "quality" of a geo assignment.
    It is defined as the minimum silhouette score for the treatment groups in
    the assignment. For more information see:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html
  3. If applied, a permutation test is used to calculate a p-value to indicate
    the likelihood of the assignment being as representative as a random
    assignment.
  A good assignment will have a low silhouette score, ideally lower than 0. A
  high score indicates that the treatment groups are separate clusters and are
  therefore not representative of the entire population.
  Example usage:
  ```
    scorer = GeoAssignmentRepresentivenessScorer(
        historical_data=historical_data,
        geo_column_name="geo_id",
        geos=geos,
        silhouette_sample_size=100,
    )
    assignment = np.array([-1, 0, 0, 1, 1, 2, 2])
    # Fast calculation without a p-value.
    score, _ = scorer(assignment)
    # Slow calculation with a p-value.
    score, pvalue = scorer(assignment, with_pvalue=True)
  ```
  """

  def __init__(
      self,
      historical_data: pd.DataFrame,
      geo_column_name: str,
      geos: list[str],
      silhouette_sample_size: int | None = None,
  ):
    """Initializes the RepresentivenessScorer.

    Args:
      historical_data: The data to score. Must contain the geo column and the
        performance metrics. All numeric columns will be included in the
        representiveness scoring.
      geo_column_name: The column name of the geo column.
      geos: The geos to be included in the assignment. If None, all geos in the
        historical data will be included. The order is important, the order of
        the assignment must match the order of the geos.
      silhouette_sample_size: The sample size to use for the silhouette score.
        If None, the full data is used.
    """
    self.historical_data = historical_data
    self.geo_column_name = geo_column_name
    self.silhouette_sample_size = silhouette_sample_size
    self.geos = geos

  @functools.cached_property
  def distance_matrix(self) -> np.array:
    """The distance matrix for the geos.

    This is calculated as 1 - the distance correlation coefficient. For more
    information see: https://arxiv.org/pdf/0803.4101
    """
    distance_matrix = np.zeros((len(self.geos), len(self.geos)))
    for i, geo_i in enumerate(self.geos):
      for j, geo_j in enumerate(self.geos):
        if i < j:
          x = (
              self.historical_data.loc[
                  self.historical_data[self.geo_column_name] == geo_i
              ]
              .select_dtypes("number")
              .values
          )
          y = (
              self.historical_data.loc[
                  self.historical_data[self.geo_column_name] == geo_j
              ]
              .select_dtypes("number")
              .values
          )
          x_normed = (x - x.mean(axis=0)) / x.std(axis=0)
          y_normed = (y - y.mean(axis=0)) / y.std(axis=0)
          corr = dcor.distance_correlation(x_normed, y_normed)
          distance_matrix[i, j] = 1.0 - corr
          distance_matrix[j, i] = 1.0 - corr
    return distance_matrix

  def _permutation_samples(
      self, assignment: np.ndarray, n_samples: int
  ) -> np.ndarray:
    """Returns the permutation samples of the score."""
    assignment = assignment.copy()
    scores = np.empty(n_samples)
    for i in range(n_samples):
      np.random.shuffle(assignment)
      scores[i] = self._score(assignment)
    return scores

  def _score(self, assignment: np.ndarray) -> float:
    """Calculates the silhouette score for the given assignment.

    This is the minimum silhouette score for the treatment groups in the
    assignment.

    Args:
      assignment: The geo assignment to score. This is assumed to be a numpy
        array of integers, where -1 indicates an excluded geo, 0 indicates a
        control geo and 1+ indicates a treatment geo (there can be multiple
        treatment groups in a multi-cell test).

    Returns:
      The silhouette score for the given assignment.
    """
    max_assignment = np.max(assignment)
    scores = []
    for i in range(1, max_assignment + 1):
      treatment_vs_rest_assignment = assignment == i
      scores.append(
          silhouette_score(
              self.distance_matrix,
              treatment_vs_rest_assignment,
              metric="precomputed",
              sample_size=self.silhouette_sample_size,
          )
      )
    return np.min(scores)

  def __call__(
      self,
      assignment: np.ndarray,
      with_pvalue: bool = False,
      n_permutation_samples: int = 200,
  ) -> tuple[float, float | None]:
    """Scores the representativeness of the given geo assignment.

    Args:
      assignment: The geo assignment to score. This is assumed to be a numpy
        array of integers, where -1 indicates an excluded geo, 0 indicates a
        control geo and 1+ indicates a treatment geo (there can be multiple
        treatment groups in a multi-cell test).
      with_pvalue: If True, a p-value will be calculated using permutation
        testing. Note: this will make the scoring much slower.
      n_permutation_samples: The number of permutation samples to use for the
        permutation test.

    Returns:
      The score of the geo assignment and the p-value if with_pvalue is True,
      otherwise None.
    """
    score = self._score(assignment)
    if with_pvalue:
      permutation_scores = self._permutation_samples(
          assignment, n_permutation_samples
      )
      pvalue = np.mean(score > permutation_scores)
    else:
      pvalue = None
    return score, pvalue


def calculate_minimum_detectable_effect_from_stats(
    standard_error: float,
    alternative: str = "two-sided",
    power: float = 0.8,
    alpha: float = 0.05,
) -> float:
  """Calculates the minimum detectable effect from the standard error.

  The minimum detectable effect is the smallest true effect size that would
  return a statistically significant result, at the specified alpha level
  and with the specified alternative hypothesis, [power]% of the time.

  The returned minimum detectable effect is always positive, even if the
  alternative hypothesis is "less".

  Args:
    standard_error: The standard error of the test statistic.
    alternative: The alternative hypothesis being tested, one of ['two-sided',
      'greater', 'less'].
    power: The desired statistical power, as a fraction. Defaults to 0.8, which
      would mean a power of 80%.
    alpha: The alpha level of the test, defaults to 0.05.

  Returns:
    The minimum detectable absolute effect size.

  Raises:
     ValueError: If the alternative is not one of ['two-sided', 'greater',
      'less'].
  """
  if alternative == "two-sided":
    z_alpha = stats.norm.ppf(q=1.0 - alpha / 2.0)
  elif alternative in ["greater", "less"]:
    z_alpha = stats.norm.ppf(q=1.0 - alpha)
  else:
    raise ValueError(
        "Alternative must be one of ['two-sided', 'greater', 'less']"
    )

  z_power = stats.norm.ppf(power)

  return standard_error * (z_alpha + z_power)
