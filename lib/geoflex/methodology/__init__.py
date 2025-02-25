"""All the individual geo-experimentation methodologies used by GeoFleX."""

import enum
from geoflex.methodology import _base
from geoflex.methodology import rct


Methodology = _base.Methodology


class MethodologyName(enum.StrEnum):
  """Methods for designing geoflex experiments.

  Note: RCT (Randomized Controlled Trial) is not a recommended methodology for
  geoflex experiments. It is included here for completeness, and for testing.
  """

  TBR_MM = "TBR_MM"  # Time Based Regression Matched Markets
  TBR = "TBR"  # Time Based Regression (No Matching)
  TM = "TM"  # Trimmed Match
  GBR = "GBR"  # Geo Based Regression
  RCT = "RCT"  # Randomized Controlled Trial.


_METHODOLOGIES = {
    MethodologyName.RCT: rct.RCT,
}


def get_methodology(methodology_name: str) -> Methodology:
  """Returns the methodology with the given name."""
  return _METHODOLOGIES[methodology_name]()
