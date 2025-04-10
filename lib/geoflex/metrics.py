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
    cost_column: The name of the column in the data that contains the cost.
  """

  name: str
  column: str = ""
  metric_per_cost: bool = False
  cost_per_metric: bool = False
  cost_column: str = ""

  model_config = pydantic.ConfigDict(extra="forbid")

  @pydantic.model_validator(mode="before")
  @classmethod
  def override_column_with_name_if_not_set(
      cls, values: dict[str, Any]
  ) -> dict[str, Any]:
    if not values.get("column"):
      values["column"] = values["name"]
    return values

  @pydantic.model_validator(mode="after")
  def check_metric_per_cost_and_cost_per_metric_are_mutually_exclusive(
      self,
  ) -> "Metric":
    if self.metric_per_cost and self.cost_per_metric:
      raise ValueError(
          "Metric cannot be both a metric per cost and a cost per metric."
      )
    return self

  @pydantic.model_validator(mode="after")
  def check_cost_column_is_set_if_metric_per_cost_or_cost_per_metric(
      self,
  ) -> "Metric":
    if self.metric_per_cost or self.cost_per_metric:
      if not self.cost_column:
        raise ValueError(
            "Cost column must be set if metric is a metric per cost or a cost"
            " per metric."
        )
    return self

  def invert(self) -> "Metric":
    """Converts a cost per metric to a metric per cost and vice versa.

    This is used for power calculations of cost per metric metrics,
    because in the null hypothesis the cost per metric is undefined (since the
    metric incrementality is zero). Therefore we need to first invert it,
    then calculate the power, and then invert it back.

    Returns:
      The inverted metric.

    Raises:
      ValueError: If the metric is not a metric per cost or a cost per metric.
    """
    inverted_suffix = " __INVERTED__"
    if self.metric_per_cost or self.cost_per_metric:
      if self.name.endswith(inverted_suffix):
        new_name = self.name.removesuffix(inverted_suffix)
      else:
        new_name = self.name + inverted_suffix

      return self.model_copy(
          update={
              "name": new_name,
              "metric_per_cost": not self.metric_per_cost,
              "cost_per_metric": not self.cost_per_metric,
          }
      )
    else:
      raise ValueError(
          "Metric cannot be inverted because it is not a metric per cost or a"
          " cost per metric."
      )


class ROAS(Metric):
  """The return of advertising spend."""

  def __init__(
      self, revenue_column: str = "revenue", cost_column: str = "cost"
  ):
    super().__init__(
        name="ROAS",
        column=revenue_column,
        metric_per_cost=True,
        cost_column=cost_column,
    )


class CPA(Metric):
  """The cost per acquisition."""

  def __init__(
      self, conversions_column: str = "conversions", cost_column: str = "cost"
  ):
    super().__init__(
        name="CPA",
        column=conversions_column,
        cost_per_metric=True,
        cost_column=cost_column,
    )
