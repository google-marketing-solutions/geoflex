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

  def __init__(self, name: str, **data: dict[str, Any]) -> None:
    if not data.get("column"):
      data["column"] = name
    super().__init__(name=name, **data)


class ROAS(Metric):
  """The return of advertising spend."""

  def __init__(self, column: str = "revenue"):
    super().__init__(name="ROAS", column=column, metric_per_cost=True)


class CPA(Metric):
  """The cost per acquisition."""

  def __init__(self, column: str = "conversions"):
    super().__init__(name="CPA", column=column, cost_per_metric=True)
