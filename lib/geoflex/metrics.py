"""The metrics for a GeoFleX experiment."""

from typing import Any
import pydantic


class Metric(pydantic.BaseModel):
  """A metric for a GeoFleX experiment.

  Attributes:
    name: The name of the metric.
    column: The name of the column in the data that contains the metric.
    metric_per_cost: Whether the metric is a metric per cost. If true, then the
      metric is calculated as the metric divided by the cost.
    cost_per_metric: Whether the metric is a cost per metric. If true, then the
      metric is calculated as the cost divided by the metric.
  """

  name: str
  column: str = ""
  metric_per_cost: bool = False
  cost_per_metric: bool = False

  model_config = pydantic.ConfigDict(extra="forbid")

  @pydantic.model_validator(mode="before")
  @classmethod
  def override_column_with_name_if_not_set(
      cls, values: dict[str, Any]
  ) -> dict[str, Any]:
    if not values.get("column"):
      values["column"] = values["name"]
    return values


class ROAS(Metric):
  """The return of advertising spend."""

  def __init__(self, column: str = "revenue"):
    super().__init__(name="ROAS", column=column, metric_per_cost=True)


class CPA(Metric):
  """The cost per acquisition."""

  def __init__(self, column: str = "conversions"):
    super().__init__(name="CPA", column=column, cost_per_metric=True)
