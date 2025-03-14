"""A design for a GeoFleX experiment."""

import enum
from typing import Any
import uuid
import geoflex.metrics
import pydantic

Metric = geoflex.metrics.Metric


class ExperimentType(enum.StrEnum):
  GO_DARK = "go_dark"
  HEAVY_UP = "heavy_up"
  HOLD_BACK = "hold_back"


class GeoAssignment(pydantic.BaseModel):
  """The geo assignment for a geoflex experiment.

  Attributes:
    control: A list of geos in the control group.
    treatment: A list of lists of geos. Each sublist represents a different
      treatment arm.
    exclude: A list of geos to exclude from the experiment.
  """

  control: list[str] = []
  treatment: list[list[str]] = []
  exclude: list[str] = []

  model_config = pydantic.ConfigDict(extra="forbid")

  @pydantic.model_validator(mode="after")
  def check_geos_not_in_multiple_groups(
      self,
  ) -> "ExperimentDesignConstraints":
    all_geo_lists = [self.control, self.exclude] + self.treatment
    seen_geos = set()
    for geo_list in all_geo_lists:
      if set(geo_list) & seen_geos:
        raise ValueError(
            "There are overlapping geos. Make sure each geo is only used once."
        )
      seen_geos.update(geo_list)

    return self


class ExperimentDesignConstraints(pydantic.BaseModel):
  """Constraints for a geoflex experiment design.

  The constraints are inputs which determine which experiment designs are
  eligible for a given experiment.

  Attributes:
    experiment_type: The type of experiment to run.
    max_runtime_weeks: The maximum number of weeks the experiment can run.
    min_runtime_weeks: The minimum number of weeks the experiment can run.
    n_cells: The number of cells to use for the experiment. Must be at least 2.
    fixed_geos: An optional subset of geos that should be fixed into a specific
      control or treatment group.
    n_geos_per_group_candidates: A list of lists of integers representing the
      number of geos per group that should be considered. The experiment design
      will choose the best configuration from this list. The inner lists must
      have the length of n_cells. If None, then the number of geos per group
      will be unconstrained.
    trimming_quantile_candidates: The candidates for the trimming quantiles to
      use for the experiment. The trimming quantile is used to trim the tails of
      the distribution of the response metric before calculating the confidence
      interval. Defaults to [0.0] (no trimming).
  """

  experiment_type: ExperimentType

  max_runtime_weeks: int = 8
  min_runtime_weeks: int = 2
  n_cells: int = 2
  fixed_geos: GeoAssignment
  n_geos_per_group_candidates: list[list[int] | None] = [None]
  trimming_quantile_candidates: list[float] = [0.0]

  model_config = pydantic.ConfigDict(extra="forbid")

  @pydantic.model_validator(mode="before")
  @classmethod
  def cast_fixed_geos(cls, values: dict[str, Any]) -> dict[str, Any]:
    if values.get("fixed_geos") is None:
      values["fixed_geos"] = GeoAssignment(
          control=[],
          treatment=[[]] * (values.get("n_cells", 2) - 1),
          exclude=[],
      )
    return values

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
  def check_n_cells_greater_than_1(
      self,
  ) -> "ExperimentDesignConstraints":
    if self.n_cells < 2:
      raise ValueError("n_cells must be greater than or equal to 2")
    return self

  @pydantic.model_validator(mode="after")
  def check_n_geos_per_group_candidates_match_n_cells(
      self,
  ) -> "ExperimentDesignConstraints":
    for candidate in self.n_geos_per_group_candidates:
      if candidate is None:
        continue
      if len(candidate) != self.n_cells:
        raise ValueError(
            "One of the candiades for the number of geos per group does not"
            " match the number of cells."
        )
    return self

  @pydantic.model_validator(mode="after")
  def check_fixed_geos_match_n_cells(
      self,
  ) -> "ExperimentDesignConstraints":
    if self.fixed_geos is not None:
      if len(self.fixed_geos.treatment) + 1 != self.n_cells:
        raise ValueError(
            "The number of fixed geo groups must match the number of cells."
        )
    return self


class ExperimentDesignEvaluation(pydantic.BaseModel):
  """The evaluation results of an experiment design."""

  design_id: str
  minimum_detectable_effects: dict[Metric, float]  # One per metric
  false_positive_rates: dict[Metric, float]  # One per metric
  power_at_minimum_detectable_effect: dict[Metric, float]  # One per metric

  model_config = pydantic.ConfigDict(extra="forbid")


class ExperimentDesign(pydantic.BaseModel):
  """An experiment design for a GeoFleX experiment.

  Attributes:
    experiment_type: The type of experiment to run.
    design_id: The unique identifier for the experiment design. This is
      autogenerated for a new design.
    primary_metric: The primary response metric for the experiment. This is the
      metric that the experiment will be designed to measure, and should be the
      main decision making metric.
    secondary_metrics: The secondary response metrics for the experiment. These
      are the metrics that the experiment will also measure, but are not as
      important as the primary metric.
    methodology: The methodology to use for the experiment.
    methodology_parameters: The parameters specific to the methodology.
    runtime_weeks: The number of weeks to run the experiment.
    n_cells: The number of cells to use for the experiment. This must be at
      least 2.
    alpha: The significance level for the experiment.
    alternative_hypothesis: The alternative hypothesis for the experiment.
    fixed_geos: An optional subset of geos that should be fixed into a specific
      control or treatment group.
    n_geos_per_group: The number of geos per group. The first group will be the
      control group, and all subsequent groups will be the treatment groups.
      Defaults to None, which means it is flexible.
    geo_assignment: The geo assignment for the experiment. This is set after the
      design is created, when the geos are assigned.
  """

  experiment_type: ExperimentType
  primary_metric: Metric
  secondary_metrics: list[Metric] = []
  methodology: str
  runtime_weeks: int
  n_cells: int = 2
  alpha: float = 0.1
  alternative_hypothesis: str = "two-sided"
  fixed_geos: GeoAssignment | None = None
  n_geos_per_group: list[int] | None = None
  methodology_parameters: dict[str, Any] = {}

  # The design id is autogenerated for a new design.
  design_id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))

  # This is set after the design is created, when the geos are assigned.
  geo_assignment: GeoAssignment | None = None

  model_config = pydantic.ConfigDict(extra="forbid")

  @property
  def pretest_weeks(self) -> int:
    """The number of weeks in the pretest period.

    If a pretest period is to be used, this should be set as the pretest_weeks
    parameter in the methodology_parameters. If this parameter is not set then
    no pretest period is used.
    """
    return self.methodology_parameters.get("pretest_weeks", 0)

  @pydantic.field_validator("primary_metric", mode="before")
  @classmethod
  def cast_metric(cls, metric: Metric | str) -> Metric:
    if isinstance(metric, str):
      return geoflex.metrics.Metric(name=metric)
    return metric

  @pydantic.field_validator("secondary_metrics", mode="before")
  @classmethod
  def cast_metrics(cls, metrics: list[Metric | str]) -> list[Metric]:
    return [
        geoflex.metrics.Metric(name=metric)
        if isinstance(metric, str)
        else metric
        for metric in metrics
    ]

  @pydantic.model_validator(mode="after")
  def check_n_cells_greater_than_1(
      self,
  ) -> "ExperimentDesignConstraints":
    if self.n_cells < 2:
      raise ValueError("n_cells must be greater than or equal to 2")
    return self

  @pydantic.model_validator(mode="after")
  def check_n_geos_per_group_matches_n_cells(
      self,
  ) -> "ExperimentDesign":
    if self.n_geos_per_group is not None:
      if len(self.n_geos_per_group) != self.n_cells:
        raise ValueError(
            "The number of geos per group does not match the number of cells."
        )
    return self

  @pydantic.model_validator(mode="after")
  def check_metric_names_do_not_overlap(
      self,
  ) -> "ExperimentDesign":
    all_metrics = [self.primary_metric] + self.secondary_metrics
    metric_names = [metric.name for metric in all_metrics]
    if len(metric_names) != len(set(metric_names)):
      raise ValueError("Metric names must be unique.")
    return self
