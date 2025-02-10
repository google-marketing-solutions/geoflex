"""Constraints for a geoflex experiment design."""

import enum
import pydantic


class ExperimentType(enum.StrEnum):
  GO_DARK = "go_dark"
  HEAVY_UP = "heavy_up"
  HOLD_BACK = "hold_back"


class Methodology(enum.StrEnum):
  """Methods for designing geoflex experiments.

  Note: RCT (Randomized Controlled Trial) is not a recommended methodology for
  geoflex experiments. It is included here for completeness, and for testing.
  """

  TBR_MM = "TBR_MM"  # Time Based Regression Matched Markets
  TBR = "TBR"  # Time Based Regression (No Matching)
  TM = "TM"  # Trimmed Match
  GBR = "GBR"  # Geo Based Regression
  RCT = "RCT"  # Randomized Controlled Trial.


class ExperimentDesignConstraints(pydantic.BaseModel):
  """Constraints for a geoflex experiment design.

  The constraints are inputs which determine which experiment designs are
  eligible for a given experiment.
  """

  experiment_type: str = ExperimentType
  eligible_methodologies: list[str] = [
      Methodology.TBR_MM,
      Methodology.TBR,
      Methodology.TM,
      Methodology.GBR,
  ]

  max_runtime_weeks: int = 8
  min_runtime_weeks: int = 2
  fixed_treatment_geos: list[str] = []
  fixed_control_geos: list[str] = []

  @pydantic.model_validator(mode="after")
  def check_max_runtime_greater_than_min_runtime(
      self,
  ) -> "ExperimentDesignConstraints":
    if self.max_runtime_weeks < self.min_runtime_weeks:
      raise ValueError(
          "max_runtime_weeks must be greater than or equal to min_runtime_weeks"
      )
    return self

  @pydantic.model_validator(mode="after")
  def check_geos_not_in_both_fixed_treatment_and_fixed_control(
      self,
  ) -> "ExperimentDesignConstraints":
    geos_in_both = set(self.fixed_treatment_geos) & set(self.fixed_control_geos)
    if geos_in_both:
      raise ValueError(
          "Geos cannot be in both fixed_treatment_geos and fixed_control_geos:"
          f" {geos_in_both}"
      )
    return self
