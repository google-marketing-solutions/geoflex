"""The Synthetic Controls methodology for GeoFleX."""

import geoflex.data
import geoflex.experiment_design
from geoflex.methodology import _base
import numpy as np
import pandas as pd


ExperimentDesign = geoflex.experiment_design.ExperimentDesign
ExperimentDesignSpec = geoflex.experiment_design.ExperimentDesignSpec
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesignEvaluation = (
    geoflex.experiment_design.ExperimentDesignEvaluation
)
GeoAssignment = geoflex.experiment_design.GeoAssignment
CellVolumeConstraintType = geoflex.experiment_design.CellVolumeConstraintType

register_methodology = _base.register_methodology


@register_methodology
class SyntheticControls(_base.Methodology):
  """The Synthetic Control methodology for GeoFleX.

  The methodology uses a synthetic control linear model to predict the
  counterfactual of the test geos based on the control geos.

  Design:
    Geos are split into treatment and control groups based on the
    maximization of predictive power of the test by the control.

  Evaluation:
    The evaluation is done with a t-test based on the difference between
    the prediction and actual values.
  """

  def assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
      rng: np.random.Generator,
  ) -> GeoAssignment:
    """Assigns geos to control and test based on the synthetic controls method.

    Args:
      experiment_design: The experiment design to assign geos for.
      historical_data: The historical data for the experiment. Can be used to
        choose geos that are similar to geos that have been used in the past.
      rng: The random number generator to use for randomization, if needed.

    Returns:
      A GeoAssignment object containing the lists of geos for the control and
      treatment groups, and optionally a list of geos that should be ignored.
    """
    pass

  def analyze_experiment(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_design: ExperimentDesign,
      experiment_start_date: str,
  ) -> pd.DataFrame:
    """Analyzes a Synthetic Control experiment.

    Returns a dataframe with the analysis results. Each row represents each
    metric provided in the experiment data. The columns are the following:

    - metric: The metric name.
    - point_estimate: The point estimate of the treatment effect.
    - lower_bound: The lower bound of the confidence interval.
    - upper_bound: The upper bound of the confidence interval.
    - point_estimate_relative: The relative effect size of the treatment.
    - lower_bound_relative: The relative lower bound of the confidence
      interval.
    - upper_bound_relative: The relative upper bound of the confidence
      interval.
    - p_value: The p-value of the null hypothesis.
    - is_significant: Whether the null hypothesis is rejected.

    Args:
      runtime_data: The runtime data for the experiment.
      experiment_design: The design of the experiment being analyzed.
      experiment_start_date: The start date of the experiment.

    Returns:
      A dataframe with the analysis results.
    """
    pass
