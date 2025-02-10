"""The Randomized Controlled Trial (RCT) methodology for GeoFleX."""

from geoflex.methodology import _base


class RCT(_base.Methodology):
  """The Randomized Controlled Trial (RCT) methodology for GeoFleX.

  It is a very simple methodolgy based on a simple A/B test. It is not
  recommended for most experiments, but can be used as a baseline for
  comparison.

  Design:
    Geos are split randomly into treatment and control groups. The treatment
    group will receive the treatment, and the control group will not. We will
    try multiple random splits and pick the best one with the lowest variance.

  Evaluation:
    The evaluation is done with a simple t-test on each test statistic.
  """
