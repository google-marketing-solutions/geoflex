"""Module containing functions for visualizing experiment analysis results."""

import pandas as pd
from pandas.io.formats import style


def _format_relative_effect_ci_string(
    row: pd.Series, decimal_places: int = 2
) -> str:
  """Used for displaying the results."""
  y_lb = row["lower_bound_relative"]
  y_ub = row["upper_bound_relative"]

  if row["point_estimate_relative"] is pd.NA:
    return ""

  return f"{{y_lb:+.{decimal_places}%}} to {{y_ub:+.{decimal_places}%}}".format(
      y_lb=y_lb, y_ub=y_ub
  )


def _format_absolute_effect_ci_string(
    row: pd.Series, significant_figures: int = 3
) -> str:
  """Used for displaying the results."""
  y_lb = row["lower_bound"]
  y_ub = row["upper_bound"]

  return (
      f"{{y_lb:+,.{significant_figures}g}} to"
      f" {{y_ub:+,.{significant_figures}g}}".format(y_lb=y_lb, y_ub=y_ub)
  )


def _format_p_value(row: pd.Series) -> str:
  """Used for displaying the results."""
  p_value = row["p_value"]
  if p_value < 0.001:
    output = "p<0.001"
  elif p_value < 0.01:
    output = f"p={p_value:.3f}"
  elif p_value <= 0.99:
    output = f"p={p_value:.2f}"
  else:
    output = "p>0.99"

  if row["is_significant"]:
    output += " (Significant)"

  return output


def _highlight_significant(row: pd.Series, props: str = "") -> list[str]:
  """Used for displaying the results."""
  del props  # unused

  if not row["is_significant"].values[0]:
    return ["color:grey"] * len(row.values)

  if row["point_estimate"].values[0] > 0.0:
    return ["background-color:lightgreen"] * len(row.values)
  else:
    return ["background-color:pink"] * len(row.values)


def display_analysis_results(
    analysis_results: pd.DataFrame,
    alpha: float,
    include_relative_effect: bool = True,
    include_absolute_effect: bool = True,
    relative_effect_decimal_places: int = 2,
    absolute_effect_significant_figures: int = 3,
) -> style.Styler:
  """Formats the experiment analysis results for displaying in a notebook.

  This formats the absolute and relative effect sizes, confidence
  intervals and p-values for all the results, and highlights statistically
  significant results.

  Args:
    analysis_results: The results of the experiment.
    alpha: The significance level, should be taken from the experiment design.
    include_relative_effect: Whether to include the relative effect size in the
      dataframe. Defaults to True.
    include_absolute_effect: Whether to include the absolute effect size (per
      item per week) in the dataframe. Defaults to True.
    relative_effect_decimal_places: Number of decimal places to round the
      relative effect sizes to.
    absolute_effect_significant_figures: Number of significant figures to round
      the absolute effect sizes to.

  Returns:
    The formatted and styled dataframe.
  """
  ci = 1.0 - alpha
  display_data = analysis_results.copy().set_index(["cell", "metric"])

  # Format the relative effect sizes
  if include_relative_effect:
    display_data[("Relative Effect Size", "Point Estimate")] = display_data[
        "point_estimate_relative"
    ].apply(f"{{:+.{relative_effect_decimal_places}%}}".format)
    display_data[("Relative Effect Size", f"{ci:.0%} CI")] = display_data.apply(
        lambda x: _format_relative_effect_ci_string(
            x, relative_effect_decimal_places
        ),
        axis=1,
    )

  # Format the absolute effect sizes
  if include_absolute_effect:
    display_data[("Absolute Effect Size", "Point Estimate")] = display_data[
        "point_estimate"
    ].apply(f"{{:+,.{absolute_effect_significant_figures}g}}".format)
    display_data[("Absolute Effect Size", f"{ci:.0%} CI")] = display_data.apply(
        lambda x: _format_absolute_effect_ci_string(
            x, absolute_effect_significant_figures
        ),
        axis=1,
    )

  # Format the p-values
  display_data[("P-value", "")] = display_data.apply(_format_p_value, axis=1)

  # Ensure all columns are two levels
  display_data.columns = pd.MultiIndex.from_tuples(
      [
          column if isinstance(column, tuple) else (column, "")
          for column in display_data.columns
      ],
      names=[" ", " "],
  )

  # Hide the irrelevant columns that already existed in the raw data
  hide_columns = [
      (column, "")
      for column in analysis_results.columns
      if column not in ["cell", "metric"]
  ]

  # Highlight the cell being hovered over
  cell_hover = {
      "selector": "td:hover",
      "props": [("background-color", "#ffffb3")],
  }

  # Format the names of the index
  index_names = {
      "selector": ".index_name",
      "props": (
          "font-style: italic; color: darkgrey; font-weight:normal; text-align:"
          " center;"
      ),
  }

  table_styles = [cell_hover, index_names]
  return (
      display_data.style.hide(hide_columns, axis="columns")
      .set_table_styles(table_styles)
      .set_table_styles(
          [
              {"selector": "th.col_heading", "props": "text-align: center;"},
              {
                  "selector": "th.col_heading.level0",
                  "props": "font-size: 1.2em;",
              },
              {"selector": "td", "props": "text-align: right;"},
          ],
          overwrite=False,
      )
      .apply(_highlight_significant, axis=1)
  )
