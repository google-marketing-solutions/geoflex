"""A design for a GeoFleX experiment."""

import enum
import itertools
from typing import Any
import uuid

import geoflex.metrics
import pandas as pd
import pydantic


Metric = geoflex.metrics.Metric


class ExperimentType(enum.StrEnum):
  GO_DARK = "go_dark"
  HEAVY_UP = "heavy_up"
  HOLD_BACK = "hold_back"


class GeoEligibility(pydantic.BaseModel):
  """The geo eligibility for a geoflex experiment.

  Attributes:
    control: A set of geos eligible for the control group.
    treatment: A list of sets of geos eligible for the treatment group(s) where
      each set corresponds to a treatment arm.
    exclude: A set of geos eligible to be excluded from the experiment.
    all_geos: A list of all possible geos for the experiment.
    flexible: default is True. This means that geos not explicitly
      defined will be flexible in terms of assignment.
  """
  control: set[str] = set()
  treatment: list[set[str]] = []
  exclude: set[str] = set()
  all_geos: set[str] = set()
  flexible: bool = True

  model_config = pydantic.ConfigDict(extra="forbid")

  @property
  def data(self) -> list[pd.DataFrame]:
    """Creates DataFrame indicating geo eligibility per treatment arm.

    Returns:
      A list of DataFrames, one per treatment cell.

      It is assumed that number of treatment arms is n_cells - 1. If the user
      has specified a 3 cell experiment, it is assumed there will be 2
      treatment arms and 1 control arm. Geoeligibility for control cell is
      captured in the treatment arm dataframe so a separate dataframe is not
      created for the control cell.

      If flexible is true, geos not explicitly defined as flexible will be
      assumed to be eligible for control or treatment in all cells.

      Each DataFrame has the following columns:
        geo: The geo id.
        control: 1 if the geo is eligible for the control, 0 otherwise.
          Value is consistent across all dataframes in the list.
        treatment: 1 if the geo is eligible for treatment in that arm, 0
          otherwise. If treatment arm eligibility is flexible, value is 1 for
          all dataframes in the list.
        exclude: 1 if the geo is eligible for exclusion, 0 otherwise.
          Value is consistent across all dataframes in the list.
    """
    num_treatment_arms = len(self.treatment)
    all_treatment_geos = set(itertools.chain.from_iterable(self.treatment))
    all_defined_geos = self.control | all_treatment_geos | self.exclude

    geos_to_process = sorted(list(self.all_geos | all_defined_geos))
    rows = [[] for _ in range(num_treatment_arms)]

    for geo in geos_to_process:
      # base eligibility is same in all treatment arms
      base_is_control = 1 if geo in self.control else 0
      base_is_exclude = 1 if geo in self.exclude else 0
      is_explicitly_defined = geo in all_defined_geos

      treatment_arm_eligibility = [
          1 if geo in arm_set else 0 for arm_set in self.treatment
      ]

      flexible_control = 0
      flexible_treatment = 0
      # if flexible is true, geos not explicitly defined are flexible
      if not is_explicitly_defined and self.flexible:
        flexible_control = 1
        flexible_treatment = 1

      # assign geos in each treatment arm(s)
      for arm_index in range(num_treatment_arms):
        row_data = {"geo": geo}
        row_data["control"] = base_is_control or flexible_control
        row_data["treatment"] = (
            treatment_arm_eligibility[arm_index]
            or flexible_treatment
        )
        row_data["exclude"] = base_is_exclude
        rows[arm_index].append(row_data)

    # create dataframes for each treatment arm
    dfs = []
    for arm_index in range(num_treatment_arms):
      df = pd.DataFrame(
          rows[arm_index] or [],
          columns=["geo", "control", "treatment", "exclude"]
      ).set_index("geo")
      dfs.append(df)

    return dfs


class GeoAssignment(GeoEligibility):
  """The geo assignment for the experiment."""

  @pydantic.model_validator(mode="after")
  def check_geos_not_in_multiple_groups(self) -> "GeoAssignment":
    """Checks that geos are not in multiple groups."""
    all_assignment_groups = [self.control, self.exclude] + self.treatment

    seen_geos = set()
    for group_set in all_assignment_groups:
      overlap = group_set & seen_geos
      if overlap:
        raise ValueError(
            f"The following geos are in multiple groups: {overlap}"
        )
      seen_geos.update(group_set)

    return self


class ExperimentDesignConstraints(pydantic.BaseModel):
  """Defines constraints for an experiment design.

  Attributes:
    experiment_type: The type of experiment to run.
    max_runtime_weeks: The maximum number of weeks the experiment can run.
    min_runtime_weeks: The minimum number of weeks the experiment can run.
    n_cells: The number of cells to use for the experiment. Must be at least 2.
    n_geos_per_group_candidates: A list of lists of integers representing the
      number of geos per group that should be considered. The experiment design
      will choose the best configuration from this list. The inner lists must
      have the length of n_cells. If None, then the number of geos per group
      will be unconstrained.
    trimming_quantile_candidates: The candidates for the trimming quantiles to
      use for the experiment. The trimming quantile is used to trim the tails of
      the distribution of the response metric before calculating the confidence
      interval. Defaults to [0.0] (no trimming).
    geo_eligibility_candidates: The geo eligibility candidates for the
      experiment.
    treatment_geos_range: (Optional) Min/max number of geos per treatment group.
    control_geos_range: (Optional) Min/max number of geos for the control group.
  """
  experiment_type: ExperimentType
  max_runtime_weeks: int = 8
  min_runtime_weeks: int = 2
  n_cells: int = 2
  n_geos_per_group_candidates: list[list[int]] | None = None
  trimming_quantile_candidates: list[float] = [0.0]
  geo_eligibility_candidates: list[GeoEligibility] = [None]
  treatment_geos_range: tuple[int, int] | None = None
  control_geos_range: tuple[int, int] | None = None

  model_config = pydantic.ConfigDict(extra="forbid")

  @pydantic.model_validator(mode="before")
  @classmethod
  def cast_geo_eligibility_candidates(
      cls, values: dict[str, Any]
  ) -> dict[str, Any]:
    raw_geo_eligibility_candidates = values.get("geo_eligibility_candidates")
    n_cells = values.get("n_cells", 2)  # Default to 2 cells if not set.
    n_treatment_groups = n_cells - 1

    unconstrained_fixed_geos = GeoEligibility(
        control=[],
        treatment=[set()] * n_treatment_groups,
        exclude=[],
    )

    if raw_geo_eligibility_candidates is None:
      values["fixed_geo_candidates"] = [unconstrained_fixed_geos.model_copy()]
    else:
      values["fixed_geo_candidates"] = [
          fixed_geos
          if fixed_geos is not None
          else unconstrained_fixed_geos.model_copy()
          for fixed_geos in raw_geo_eligibility_candidates
      ]
    return values

  @pydantic.model_validator(mode="after")
  def check_max_runtime_greater_than_min_runtime(
      self,
  ) -> "ExperimentDesignConstraints":
    """Checks that max_runtime_weeks is greater than or equal to min_runtime_weeks.

    Raises:
      ValueError: If max_runtime_weeks is less than min_runtime_weeks.

    Returns:
      The ExperimentDesignConstraints object.
    """
    if self.max_runtime_weeks < self.min_runtime_weeks:
      raise ValueError(
          "max_runtime_weeks must be greater than or equal to min_runtime_weeks"
      )
    return self

  @pydantic.model_validator(mode="after")
  def check_n_cells_greater_than_1(self) -> "ExperimentDesignConstraints":
    """Checks that n_cells is greater than or equal to 2.

    Raises:
        ValueError: If n_cells is less than 2.

    Returns:
        The ExperimentDesignConstraints object.
    """
    if self.n_cells < 2:
      raise ValueError("n_cells must be greater than or equal to 2")
    return self

  def check_n_geos_per_group_candidates_match_n_cells(
      self,
  ) -> "ExperimentDesignConstraints":
    """Checks if number of geos per group candidates matches number of cells.

    Raises:
        ValueError: If the number of geos per group candidates does not match
        the number of cells.

    Returns:
        The ExperimentDesignConstraints object.
    """
    if self.n_geos_per_group_candidates:
      for candidate in self.n_geos_per_group_candidates:
        if len(candidate) != self.n_cells:
          raise ValueError(
              "The number of geos per group does not match the number of cells."
          )
    return self

  @pydantic.model_validator(mode="after")
  def check_geo_eligibility_matches_n_cells(
      self
  ) -> "ExperimentDesignConstraints":
    """Checks if number of treamtment arms matches n_cells."""
    for geo_eligibility in self.geo_eligibility_candidates:
      num_treatment_arms_in_geoeligibility = len(geo_eligibility.treatment)
      expected_num_treatment_arms = self.n_cells - 1

      if num_treatment_arms_in_geoeligibility != expected_num_treatment_arms:
        raise ValueError(
            "The number of treatment arms in the geo eligibility does not match"
            " the number of cells."
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
    n_cells: The number of cells to use for the experiment.
      This must be at least 2.
    alpha: The significance level for the experiment.
    alternative_hypothesis: The alternative hypothesis for the experiment.
    n_geos_per_group: The number of geos per group. The first group will be the
      control group, and all subsequent groups will be the treatment groups.
      Defaults to None, which means it is flexible.
    constraints: The constraints for the experiment.
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
  methodology_parameters: dict[str, Any] = {}

  # The design id is autogenerated for a new design.
  design_id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))

  # Store geo assignment directly
  geo_assignment: GeoAssignment | None = None
  n_geos_per_group: list[int] | None = None

  model_config = pydantic.ConfigDict(extra="forbid")

  @property
  def n_cells(self) -> int:
    """The number of cells to use for the experiment."""
    return self.constraints.n_cells

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

  @pydantic.field_validator(mode="after")
  def check_n_geos_per_group_matches_n_cells(self) -> "ExperimentDesign":
    """Checks that the number of geos per group matches the number of cells."""
    if self.n_geos_per_group is not None:
      if len(self.n_geos_per_group) != self.n_cells:
        raise ValueError(
            f"Length of n_geos_per_group ({len(self.n_geos_per_group)}) "
            f"does not match n_cells ({self.n_cells})."
        )
    return self

  @pydantic.field_validator(mode="after")
  def check_geoassignment_matches_n_cells(self) -> "ExperimentDesign":
    """Checks if number of treatment arms in geoassignment matches n_cells."""
    if self.geo_assignment is not None:
      num_treatment_arms_in_assignment = len(
          self.geo_assignment.treatment
      )
      expected_num_treatment_arms = self.n_cells - 1
      if num_treatment_arms_in_assignment != expected_num_treatment_arms:
        raise ValueError(
            f"Expected {expected_num_treatment_arms} treatment arms "
            f"in geo_assignment (based on n_cells={self.n_cells}), "
            f"but found {num_treatment_arms_in_assignment}."
        )
    return self
