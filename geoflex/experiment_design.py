"""A design for a GeoFleX experiment."""

import enum
from typing import Any
import pydantic


class ExperimentType(enum.StrEnum):
  GO_DARK = "go_dark"
  HEAVY_UP = "heavy_up"
  HOLD_BACK = "hold_back"


class ExperimentDesignConstraints(pydantic.BaseModel):
  """Constraints for a geoflex experiment design.

  The constraints are inputs which determine which experiment designs are
  eligible for a given experiment.
  """

  experiment_type: str = ExperimentType

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


class ExperimentDesignEvaluation(pydantic.BaseModel):
  """The evaluation results of an experiment design."""

  design_id: str
  minimum_detectable_effects: dict[str, float]  # One per metric
  false_positive_rates: dict[str, float]  # One per metric
  power_at_minimum_detectable_effect: dict[str, float]  # One per metric


class GeoAssignment(pydantic.BaseModel):
  """The geo assignment for a geoflex experiment."""

  control: list[str]
  treatment: list[str]
  ignored: list[str]


class ExperimentDesign(pydantic.BaseModel):
  """An experiment design for a GeoFleX experiment."""

  design_id: str
  primary_response_metric: str
  methodology: str
  methodology_parameters: dict[str, Any]
  runtime_weeks: int
  alpha: float

  geo_assignment: GeoAssignment | None = None
