"""Time based regression matched markets (TBRMM) methodology for GeoFleX."""

import logging
from typing import Any, Optional

import geoflex.data
import geoflex.evaluation
import geoflex.experiment_design
import geoflex.exploration_spec
from geoflex.methodology import _base
import geoflex.metrics
# pylint: disable = line-too-long
from matched_markets.methodology import geoeligibility as original_geoeligibility_module
from matched_markets.methodology import tbr as original_tbr_analysis
from matched_markets.methodology import tbrmatchedmarkets as original_tbrmm_main
from matched_markets.methodology import tbrmmdata as original_tbrmmdata
from matched_markets.methodology import tbrmmdesignparameters as original_tbrmmdesignparameters
# pylint: enable = line-too-long
import numpy as np
import pandas as pd
import pydantic
from scipy import stats


GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesign = geoflex.experiment_design.ExperimentDesign
ExperimentDesignSpec = geoflex.exploration_spec.ExperimentDesignExplorationSpec
GeoAssignment = geoflex.experiment_design.GeoAssignment
GeoEligibility = geoflex.experiment_design.GeoEligibility
Metric = geoflex.experiment_design.Metric
register_methodology = _base.register_methodology

logger = logging.getLogger(__name__)


class TBRMMParameters(pydantic.BaseModel):
  """Parameters specific to the TBRMM methodology for GeoFleX.

  These are tuned by Optuna and used to construct the original library's
  TBRMMDesignParameters object.
  """
  pretest_weeks: int = pydantic.Field(
      ...,  # required
      gt=0,
      description="""Number of weeks for pretest period used for matching and
      model training."""
  )
  min_corr: Optional[float] = pydantic.Field(
      default=0.8,
      ge=0.8,
      lt=1.0,
      description="""Minimum correlation coefficient between the control and
      treatment groups."""
  )
  rho_max: Optional[float] = pydantic.Field(
      default=None,
      description="Maximum auto-correlation (rho) for the residuals."
  )
  volume_ratio_tolerance: Optional[float] = pydantic.Field(
      default=None,
      description="Tolerance for the ratio of volumes between groups."
  )
  geo_ratio_tolerance: Optional[float] = pydantic.Field(
      default=None,
      description="Tolerance for the ratio of number of geos between groups."
  )
  assumed_iroas: Optional[float] = pydantic.Field(
      default=1.0,
      description="Assumed true target iROAS for original TBRMM library."
  )
  cost_column: Optional[str] = pydantic.Field(
      default=None,
      description="The column name for cost data, used for budget constraints."
  )
  treatment_share_range: Optional[tuple[float, float]] = pydantic.Field(
      default=None,
      description="""Constrain the % of the treatment group response with
      respect to the overall response."""
  )
  treatment_geos_range: Optional[tuple[int, int]] = pydantic.Field(
      default=None,
      description="""Minimum and maximum number of geos to include in the
      treatment group."""
  )
  control_geos_range: Optional[tuple[int, int]] = pydantic.Field(
      default=None,
      description="""Minimum and maximum number of geos to include in the
      control group."""
  )
  model_config = pydantic.ConfigDict(extra="ignore")


@register_methodology
class TBRMM(_base.Methodology):
  """Time-Based Regression with Matched Markets (TBRMM) methodology for GeoFleX.

  Acts as a wrapper around the original 'google/matched_markets' library.
  Adapts GeoFleX inputs, calls original library's core algorithms for
  geo assignment (hill-climbing search) and analysis (TBR model), and adapts
  results back to GeoFleX format.

  This methodology is restricted to 2-cell designs (1 Control, 1 Treatment) and
  relies on the availability of the 'google/matched_markets' library.

  Design:
    Geo assignment is done using direct calls to the original library's
    matched market greedy search algorithm and analysis is done using
    direct calls to the original library's TBR model.

  Evaluation:
    The TBRMM methodology is evaluated through summary statistics which compare
    the TBR model's fit to the observed data.
  """
  is_pseudo_experiment = True  # enforced to True for TBRMM
  default_methodology_parameter_candidates: dict[str, list[Any]] = {
      "pretest_weeks": [4, 8, 12, 16, 26, 52],
      "min_corr": [0.8, 0.85, 0.9, 0.95],
      "rho_max": [None, 0.995],
      "volume_ratio_tolerance": [None],
      "geo_ratio_tolerance": [None],
  }

  def _methodology_is_eligible_for_design_and_data(
      self, design: ExperimentDesign, pretest_data: GeoPerformanceDataset
  ) -> bool:
    """Checks if the design is eligible for the TBRMM methodology.

    Args:
      design: The experiment design to check.
      pretest_data: The pretest data (unused in this implementation).

    Returns:
      True if the design is eligible, False otherwise.
    """
    if design.n_cells != 2:
      logger.info(
          "TBRMM methodology only supports n_cells=2. Got %s.", design.n_cells
      )
      return False
    return True

  def _methodology_assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
  ) -> tuple[GeoAssignment, dict[str, Any]]:
    """Assigns geos using the original google/matched_markets library's search.

    If the number of eligible geos is less than 10, it attempts exhaustive
    search first, falling back to greedy search if exhaustive fails or finds
    no designs. Otherwise, greedy search is used.

    Budget constraint is strict in exhaustive search but not in greedy search.
    Budget could be a reason fallback for greedy search is triggered. It is
    possible for both to return no design if geo constraints are overly
    restrictive or data does not align well with TBR model.

    Args:
      experiment_design: The experiment design to assign geos for.
      historical_data: The historical data source to assign geos for.
    Returns:
      The geo assignment and an empty dictionary for intermediate data.
    """
    method_name = experiment_design.methodology
    design_id = experiment_design.design_id
    logger.info(
        "Starting %s geo assignment for design %s using original library...",
        method_name, design_id
    )

    logger.debug(
        """Adapting GeoFleX inputs to match expected format of original TBRMM
        library so that direct calls to the original library can be made."""
    )
    try:
      geoflex_tbrmm_params = TBRMMParameters.model_validate(
          experiment_design.methodology_parameters
          )
      original_gelig_obj = self._adapt_geoflex_geo_eligibility_to_original(
          experiment_design.geo_eligibility,
          list(historical_data.geos)
          )

      cost_column = geoflex_tbrmm_params.cost_column

      original_data_obj = self._adapt_geoflex_data_to_original_tbrmmdata(
          historical_data,
          experiment_design.primary_metric.column,
          original_gelig_obj,
          cost_column=cost_column,
      )
      original_design_params_obj = (
          self._adapt_geoflex_design_to_original_params(
              experiment_design,
              geoflex_tbrmm_params,
              historical_data.data_frequency_days,
              cost_column
              )
      )
    except (
        TypeError,
        TypeError,
        ValueError,
        KeyError,
        pydantic.ValidationError,
        AttributeError,
        pd.errors.ParserError
        ) as e:
      logger.error(
          "Error adapting GeoFleX inputs to original library formats: %s",
          e, exc_info=True
          )
      raise ValueError(
          "Failed to adapt inputs for original TBRMM library.") from e

    logger.debug("Calling original library's search function...")
    best_design_from_search = None
    search_method_used_info = "N/A"
    try:
      original_tbrmm_instance = original_tbrmm_main.TBRMatchedMarkets(
          data=original_data_obj,
          parameters=original_design_params_obj
      )

      # original_data_obj.df may be pivoted.
      # original library's GeoEligibility object sets 'geo' as index
      num_geos_for_search = (
          original_data_obj.geo_eligibility.data.index.nunique()
      )
      logger.info(
          "Number of unique geos available for search: %s",
          num_geos_for_search)

      if num_geos_for_search < 10:
        search_method_used_info = "exhaustive (attempted for <10 geos)"
        logger.info(
            "Number of geos (%s) < 10. Attempting exhaustive search.",
            num_geos_for_search)
        try:
          exhaustive_results = original_tbrmm_instance.exhaustive_search()
          if exhaustive_results:
            best_design_from_search = exhaustive_results[0]
            search_method_used_info = "exhaustive (successful for <10 geos)"
            logger.info("Exhaustive search successful and returned a design.")
          else:
            logger.warning(
                """Exhaustive search returned no suitable designs.
                Will attempt greedy search.""")
        except (
            RuntimeError, ValueError, KeyError, AttributeError, TypeError
            ) as e_ex:
          logger.warning(
              "Exhaustive search failed: %s. Will attempt greedy search.",
              e_ex, exc_info=False)

        if not best_design_from_search:
          search_method_used_info = "greedy (fallback for <10 geos)"
          logger.info("""Attempting greedy search because exhaustive search
                      failed or found no design for <10 geos.""")
          try:
            greedy_results = original_tbrmm_instance.greedy_search()
            if greedy_results:
              best_design_from_search = greedy_results[0]
              logger.info("Greedy search (fallback) successful.")
            else:
              logger.warning(
                  """Greedy search (fallback for <10 geos) also returned
                  no suitable designs.""")
          except (
              RuntimeError, ValueError, KeyError, AttributeError, TypeError
              ) as e_gr_fallback:
            logger.warning(
                "Greedy search (fallback for <10 geos) also failed: %s.",
                e_gr_fallback, exc_info=False)
      # greedy search is only option if num_geos >= 10
      else:
        search_method_used_info = "greedy (geos >= 10)"
        logger.info(
            "Number of geos (%s) >= 10. Using greedy search.",
            num_geos_for_search)
        try:
          greedy_results = original_tbrmm_instance.greedy_search()
          if greedy_results:
            best_design_from_search = greedy_results[0]
            logger.info("Greedy search successful for geos >= 10.")
          else:
            logger.warning(
                "Greedy search (for geos >= 10) returned no suitable designs.")
        except (RuntimeError, ValueError) as e_gr_primary:
          logger.error(
              """Greedy search (for geos >= 10) failed with expected
               error type: %s""", e_gr_primary, exc_info=True)
          raise
        #  unexpected error leads to best_design_from_search being None
        except (KeyError, AttributeError, TypeError) as e_gr_other:
          logger.error(
              "Greedy search (for geos >= 10) failed with unexpected error: %s",
              e_gr_other, exc_info=True)
    except (RuntimeError, ValueError) as e:
      logger.error(
          """Original TBRMM library search failed during
           instantiation or primary search path: %s""", e, exc_info=True)
      raise RuntimeError(
          f"""Original TBRMM library search failed
           (strategy: {search_method_used_info}).""") from e

    logger.debug(
        """Adapting original output (from %s search)
         back to format of GeoFleX GeoAssignment...""", search_method_used_info
        )
    if not best_design_from_search:
      raise ValueError(
          f"""Original TBRMM library's search
           (strategy: {search_method_used_info}) returned
           no suitable designs."""
          )
    original_best_design = best_design_from_search

    # greedy_search doesn't have a final budget check
    # manually perform budget check and log
    budget_range = original_design_params_obj.budget_range
    if budget_range is not None:
      if original_best_design.diag is None:
        logger.warning(
            "Cannot check budget constraint; returned design is missing the"
            " diagnostics object."
        )
      else:
        try:
          required_impact = original_best_design.diag.required_impact
          iroas = original_design_params_obj.iroas
          if iroas > 0:
            required_budget = required_impact / iroas
            logger.info(
                "Selected design has an estimated cost of %.2f (impact: %.2f /"
                " iroas: %.2f). Budget range: %s",
                required_budget, required_impact, iroas, budget_range
            )
            # check if design is within budget
            if not (budget_range[0] <= required_budget <= budget_range[1]):
              # get GeoFleX budget info for the warning message
              geoflex_budget = experiment_design.experiment_budget
              geoflex_budget_type = geoflex_budget.budget_type
              geoflex_budget_value = (
                  geoflex_budget.value[0]
                  if isinstance(geoflex_budget.value, list)
                  else geoflex_budget.value
              )
              budget_details = (
                  f"{geoflex_budget_type.value}: {geoflex_budget_value}"
              )
              if (geoflex_budget_type ==
                  geoflex.experiment_design.ExperimentBudgetType.DAILY_BUDGET):
                total_budget = (
                    geoflex_budget_value * experiment_design.runtime_weeks * 7
                )
                budget_details += (
                    f" (transformed to total budget of {total_budget:.2f} for"
                    f" {experiment_design.runtime_weeks} weeks)"
                )

              warning_message = (
                  f"The best design found by the '{search_method_used_info}'"
                  " search is over budget. The design will be used, but please"
                  " proceed with caution. GeoFleX Budget Input:"
                  f" {budget_details}. Estimated cost: {required_budget:.2f}."
                  f" Allowed budget range: {budget_range}"
              )
              logger.warning(warning_message)
          else:
            logger.warning("Cannot check budget: iROAS is not positive.")
        except AttributeError:
          logger.warning(
              "Could not check budget or estimate cost: 'diag.required_impact'"
              " attribute not found in the original design object."
          )

    geoflex_assignment = (
        self._adapt_original_search_result_to_geoflex_assignment(
            original_best_design,
            set(historical_data.geos)
            )
    )

    logger.info(
        """Completed geo assignment using original library for
         design %s (method: %s).""", design_id, search_method_used_info
        )
    # return geo assignment and empty dict for intermediate data
    return geoflex_assignment, {}

  def _methodology_analyze_experiment(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_design: ExperimentDesign,
      experiment_start_date: pd.Timestamp,
      experiment_end_date: pd.Timestamp,
      pretest_period_end_date: pd.Timestamp,
  ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Analyzes 2-cell TBRMM experiment using original TBRMM library.

    Adapts data once for all metrics and then fits the TBR model for each.
    Uses the geo assignment provided in the experiment_design object.

    Args:
      runtime_data: The runtime data to analyze. Data is bootstrapped during
        evaluation step.
      experiment_design: The experiment design to analyze (contains the
        assignment to use).
      experiment_start_date: The start date of the experiment.
      experiment_end_date: The end date of the experiment.
      pretest_period_end_date: The end date of the pretest period.
    Returns:
      The analysis results as a DataFrame and an empty dictionary for
      intermediate data.
    """
    method_name = experiment_design.methodology
    design_id = experiment_design.design_id
    logger.info(
        "Starting %s analysis for design %s using original library calls...",
        method_name, design_id
        )

    if experiment_design.n_cells != 2:
      raise NotImplementedError(
          f"{method_name} analysis only supports 2-cell designs."
          )

    if experiment_design.geo_assignment is None:
      raise ValueError(
          "GeoAssignment missing in ExperimentDesign for analysis."
          )

    try:
      geoflex_tbrmm_params = TBRMMParameters.model_validate(
          experiment_design.methodology_parameters
          )
      alpha = experiment_design.alpha
      # fix_assignment parameter is handled by the evaluator
    except pydantic.ValidationError as e:
      raise ValueError(
          f"Invalid {method_name} parameters for analysis: %s") from e

    # Use the geo assignment provided in the experiment_design object.
    # The evaluator ensures this is the correct assignment (fixed or re-run).
    assignment_to_use = experiment_design.geo_assignment

    # Ensure the assignment has control and treatment geos
    if (not assignment_to_use.control
        or not assignment_to_use.treatment
        or not assignment_to_use.treatment[0]):
      raise ValueError(
          "Control or treatment geos are missing in the GeoAssignment."
          )

    all_results_list = []
    all_metrics: list[Metric] = (
        [experiment_design.primary_metric] +
        experiment_design.secondary_metrics
        )
    # Collect all metric column names
    all_metric_column_names = [metric.column for metric in all_metrics]

    # Adapt data once for all metrics using the provided assignment
    try:
      adapted_long_format_df, semantic_kwargs = (
          self._adapt_geoflex_data_to_original_tbr_input(
              runtime_data=runtime_data,
              assignment=assignment_to_use,
              metric_column_names=all_metric_column_names,
              pretest_weeks=geoflex_tbrmm_params.pretest_weeks,
              experiment_start_date=experiment_start_date,
              experiment_end_date=experiment_end_date,
              date_col_name=runtime_data.date_column,
              geo_col_name=runtime_data.geo_id_column,
              )
          )
      if adapted_long_format_df.empty:
        raise ValueError(
            "Adapted data for original TBR.fit() is empty."
            )

    except(
        ValueError,
        KeyError,
        AttributeError,
        RuntimeError,
        TypeError,
        IndexError,
        pd.errors.ParserError) as e:
      logger.error(
          "Data adaptation failed for original TBR library: %s",
          e, exc_info=True
          )
      raise RuntimeError(
          "Failed to adapt data for original TBR library.") from e

    for metric_obj in all_metrics:
      metric_column_name = metric_obj.column
      logger.debug(
          "Analyzing metric: %s (%s) via original library",
          metric_obj.name, metric_column_name
          )
      results_dict = self._get_empty_analysis_result_row(
          cell_number=1, metric=metric_obj, design=experiment_design
          )
      try:
        # Select the target metric column from the already adapted DataFrame
        data_for_tbr_fit = adapted_long_format_df[[
            semantic_kwargs["key_date"],
            semantic_kwargs["key_group"],
            semantic_kwargs["key_period"],
            metric_column_name  # Select the specific metric column
        ]].copy()

        original_tbr_analyzer = original_tbr_analysis.TBR(use_cooldown=False)
        original_tbr_analyzer.fit(
            data_frame=data_for_tbr_fit,  # DataFrame sliced for this metric
            target=metric_column_name,
            **semantic_kwargs)
        summary_df_original = original_tbr_analyzer.summary(
            report="last",
            level=(1.0 - alpha),
            tails=2 if (
                experiment_design.alternative_hypothesis == "two-sided") else 1
            )
        if summary_df_original.empty:
          raise ValueError(
              "Original TBR.summary() returned empty for"
              f"metric {metric_obj.name}.")

        # Calculate control total runtime for the current metric
        control_total_runtime = self._get_control_total_for_relative(
            data_for_tbr_fit,  # DataFrame sliced for this metric
            metric_column_name,
            semantic_kwargs
            )

        results_dict.update(self._adapt_original_tbr_summary_to_geoflex(
            original_summary_row=summary_df_original.iloc[0],
            control_total_runtime=control_total_runtime,
            alternative_hypothesis=experiment_design.alternative_hypothesis,
            alpha=alpha,
            metric_name=metric_obj.name
            ))
      except(
          ValueError,
          KeyError,
          AttributeError,
          RuntimeError,
          TypeError,
          IndexError,
          pd.errors.ParserError) as e:
        logger.warning(
            "Analysis failed for metric '%s' using original library: %s",
            metric_obj.name, e, exc_info=True
            )

      all_results_list.append(results_dict)

    logger.info(
        "Completed %s analysis using original library for design '%s'.",
        method_name, design_id
        )
    # return the DataFrame and an empty dict for intermediate data
    return pd.DataFrame(all_results_list), {}

  # --- Adapter Helper Methods ---
  def _adapt_original_search_result_to_geoflex_assignment(
      self,
      original_best_design: Any,
      all_dataset_geos: set[str]
  ) -> GeoAssignment:
    """Adapts the best geo assignment from the original library to GeoFleX.

    Args:
      original_best_design: The best design object from the original
        library's search results.
      all_dataset_geos: A set of all geo IDs present in the input dataset.

    Returns:
      A GeoFleX GeoAssignment object.

    Raises:
      ValueError: If the original design object is malformed or attributes are
        missing, or if adaptation fails.
    """
    logger.debug(
        "Adapting original search result to GeoFleX GeoAssignment."
    )
    try:
      control_set = set(original_best_design.control_geos)
      treatment_set = set(original_best_design.treatment_geos)
      assigned_geos = control_set | treatment_set
      exclude_set = all_dataset_geos - assigned_geos
      return geoflex.experiment_design.GeoAssignment(
          control=control_set,
          treatment=[treatment_set],  # GeoFleX expects a list of treatment sets
          exclude=exclude_set
      )
    except AttributeError as e:
      logger.error(
          "Original design object from search result missing attributes: %s",
          e, exc_info=True
      )
      raise ValueError(
          "Could not parse geo assignment output from original library."
      ) from e
    except (ValueError, TypeError, KeyError) as e:
      logger.error(
          "Error converting original output to GeoFleX GeoAssignment: %s",
          e, exc_info=True
      )
      raise ValueError(
          "Failed to adapt original search result to GeoAssignment."
      ) from e

  def _adapt_geoflex_geo_eligibility_to_original(
      self,
      geoflex_eligibility: GeoEligibility,
      all_data_geos: list[str]
  ) -> original_geoeligibility_module.GeoEligibility:
    """Converts GeoFleX GeoEligibility to work for original library.

    DataFrame-based GeoEligibility object is expected by the original
    google/matched_markets library. GeoFleX's GeoEligibility Pydantic model
    is converted to a DataFrame and then to the original library's
    GeoEligibility object.

    Args:
      geoflex_eligibility: GeoFleX's GeoEligibility Pydantic model. Assumed
        to be non-None, as ExperimentDesign replaces None with an empty object.
      all_data_geos: List of all geo strings in the current dataset.

    Returns:
      An instance of the original library's GeoEligibility class.
    """
    logger.debug(
        """Converting GeoFleX GeoEligibility to original format to
        enable direct calls to the original library."""
        )
    # if geoflex_eligibility.data is empty
    if not geoflex_eligibility.data:
      logger.warning(
          """GeoFleX GeoEligibility.data property returned an empty list
          or GeoEligibility was initially empty. Defaulting to flexible.""")
      data_for_df = [{"geo": geo, "control": 1, "treatment": 1, "exclude": 1}
                     for geo in all_data_geos]
      adapted_df = pd.DataFrame(data_for_df)
      if adapted_df.empty:
        adapted_df = pd.DataFrame(
            columns=["geo", "control", "treatment", "exclude"]
            )

    else:
      geoflex_eligibility_df = geoflex_eligibility.data[0].copy()
      geos_in_elig_df = set(geoflex_eligibility_df.index)
      geos_from_dataset = set(all_data_geos)
      missing_from_elig_df = geos_from_dataset - geos_in_elig_df

      if missing_from_elig_df:
        added_rows_df: pd.DataFrame
        if geoflex_eligibility.flexible:
          new_rows_data = []
          for geo in missing_from_elig_df:
            new_rows_data.append(
                {"geo": geo, "control": 1, "treatment": 1, "exclude": 1}
                )
          added_rows_df = pd.DataFrame(new_rows_data).set_index("geo")
        # if geoflex_eligibility.flexible is False
        else:
          # geos in the dataset but not defined due to inflexible arg
          # must be marked as excluded
          new_rows_data = []
          for geo in missing_from_elig_df:
            new_rows_data.append(
                {"geo": geo, "control": 0, "treatment": 0, "exclude": 1}
            )
          added_rows_df = pd.DataFrame(new_rows_data).set_index("geo")

        geoflex_eligibility_df = pd.concat(
            [geoflex_eligibility_df, added_rows_df]
        )

      adapted_df = geoflex_eligibility_df.reset_index()

    required_cols = ["geo", "control", "treatment", "exclude"]
    for col in required_cols:
      if col not in adapted_df.columns:
        raise ValueError(
            f"Column '{col}' missing from GeoFleX GeoEligibility.data.")

    return original_geoeligibility_module.GeoEligibility(adapted_df)

  def _adapt_geoflex_data_to_original_tbrmmdata(
      self,
      geoflex_data_source: GeoPerformanceDataset,
      metric_column: str,
      original_gelig_obj: original_geoeligibility_module.GeoEligibility,
      cost_column: Optional[str] = None,
  ) -> original_tbrmmdata.TBRMMData:
    """Converts GeoFleX GeoPerformanceDataset for the original library.

    The original google/matched_markets library expects a TBRMMData object
    which contains specific columns and column types.

    Args:
      geoflex_data_source: GeoPerformanceDataset object containing the data.
      metric_column: The column name of the metric to use in the TBRMM analysis.
      original_gelig_obj: The original GeoEligibility object.
      cost_column: The column name of the cost data to use for budgeting.

    Raises:
      KeyError: If the metric column is not found in the GeoPerformanceDataset.

    Returns:
      An instance of the original library's TBRMMData class.
    """
    logger.debug(
        "Adapting GeoFleX GeoPerformanceDataset to original TBRMMData format."
    )
    gf_geo_col = geoflex_data_source.geo_id_column
    gf_date_col = geoflex_data_source.date_column
    columns_to_select = {gf_geo_col, gf_date_col, metric_column}
    rename_map = {
        gf_geo_col: "geo",
        gf_date_col: "date",
        metric_column: "response",
    }
    tbrmm_data_kwargs = {}

    if cost_column:
      if cost_column not in geoflex_data_source.parsed_data.columns:
        raise KeyError(
            f"Cost column '{cost_column}' not found in GeoPerformanceDataset."
        )
      columns_to_select.add(cost_column)
      rename_map[cost_column] = "cost"

    if metric_column not in geoflex_data_source.parsed_data.columns:
      raise KeyError(
          f"Metric col '{metric_column}' not found in GeoPerformanceDataset."
      )
    long_df = geoflex_data_source.parsed_data[list(columns_to_select)].copy()
    long_df.rename(columns=rename_map, inplace=True)
    long_df["geo"] = long_df["geo"].astype(str)

    tbrmm_data_kwargs.update({
        "df": long_df,
        "response_column": "response",
        "geo_eligibility": original_gelig_obj,
    })
    return original_tbrmmdata.TBRMMData(**tbrmm_data_kwargs)

  def _adapt_geoflex_design_to_original_params(
      self,
      geoflex_design_obj: ExperimentDesign,
      geoflex_tbrmm_params: TBRMMParameters,
      historical_data_freq_days: int,
      cost_column: Optional[str] = None
  ) -> original_tbrmmdesignparameters.TBRMMDesignParameters:
    """Converts GeoFleX ExperimentDesign & TBRMMParameters for original library.

    The original google/matched_markets library expects a TBRMMDesignParameters
    with specific attributes.

    Args:
      geoflex_design_obj: GeoFleX ExperimentDesign object.
      geoflex_tbrmm_params: GeoFleX TBRMMParameters object.
      historical_data_freq_days: Number of days per data point in the
        historical data (e.g., 1 for daily, 7 for weekly).
      cost_column: The name of the cost column, if available.
    Returns:
      An instance of the original library's TBRMMDesignParameters class.
    Raises:
      ValueError: If essential parameters are invalid or missing.
    """
    logger.debug(
        "Adapting GeoFleX design and methodology parameters to original"
        " library format."
        )

    if historical_data_freq_days <= 0:
      raise ValueError(
          "historical_data_freq_days must be a positive integer."
          )
    if (
        geoflex_design_obj.runtime_weeks is None
        or geoflex_design_obj.runtime_weeks <= 0
    ):
      raise ValueError(
          "ExperimentDesign.runtime_weeks must be a positive integer.")

    # calculate n_test for original library
    # number of time points in experiment period
    n_test_points = int(
        geoflex_design_obj.runtime_weeks * (7 / historical_data_freq_days)
        )
    if n_test_points < 1:  # min n_test is 1
      raise ValueError(
          f"Calculated n_test_points ({n_test_points}) is less than 1."
          " Check runtime_weeks and historical_data_freq_days."
          )

    # calculate n_pretest_max for original library
    # max number of time points in pretest period
    n_pretest_points = int(
        geoflex_tbrmm_params.pretest_weeks * (7 / historical_data_freq_days)
        )
    if n_pretest_points < 3:  # min n_pretest_max is 3
      logger.warning(
          "Calculated n_pretest_points (%s) is less than minimum of 3."
          " Adjusting to 3.", n_pretest_points
          )
      n_pretest_points = 3

    # Use the assumed_iroas from TBRMMParameters as required by the original
    assumed_iroas = geoflex_tbrmm_params.assumed_iroas
    logger.info(
        "Using assumed_iroas=%s from TBRMMParameters for original library.",
        assumed_iroas
        )

    # adapt power_level and sig_level
    # assume default power level of 0.8
    original_power_level = 0.8
    original_sig_level = 1.0 - geoflex_design_obj.alpha

    # prepare kwargs for original TBRMMDesignParameters
    # defaults from original library will be used if not set here
    original_params_kwargs = {
        "n_test": n_test_points,
        "iroas": assumed_iroas,  # required by original library
        "n_pretest_max": n_pretest_points,
        "sig_level": original_sig_level,
        "power_level": original_power_level,
        "n_designs": 1,  # GeoFleX handles multiple trials; original lib gets 1
    }

    # optional parameters from GeoFleX TBRMMParameters
    # if None in geoflex_tbrmm_params, original library's defaults will apply
    # will remain None if the original library treats as no constraint
    if geoflex_tbrmm_params.min_corr is not None:
      original_params_kwargs["min_corr"] = geoflex_tbrmm_params.min_corr

    if geoflex_tbrmm_params.rho_max is not None:
      original_params_kwargs["rho_max"] = geoflex_tbrmm_params.rho_max

    if geoflex_tbrmm_params.volume_ratio_tolerance is not None:
      original_params_kwargs[
          "volume_ratio_tolerance"
          ] = geoflex_tbrmm_params.volume_ratio_tolerance

    if geoflex_tbrmm_params.geo_ratio_tolerance is not None:
      original_params_kwargs[
          "geo_ratio_tolerance"
          ] = geoflex_tbrmm_params.geo_ratio_tolerance

    if geoflex_tbrmm_params.treatment_share_range is not None:
      original_params_kwargs["treatment_share_range"] = (
          geoflex_tbrmm_params.treatment_share_range
      )
    if geoflex_tbrmm_params.control_geos_range is not None:
      original_params_kwargs["control_geos_range"] = (
          geoflex_tbrmm_params.control_geos_range
      )
    if geoflex_tbrmm_params.treatment_geos_range is not None:
      original_params_kwargs["treatment_geos_range"] = (
          geoflex_tbrmm_params.treatment_geos_range
      )

    # adapt GeoFleX budget to original library's budget_range
    budget = geoflex_design_obj.experiment_budget
    if budget:
      if not cost_column:
        logger.warning(
            "A budget is specified for the experiment, but the 'cost_column' "
            "is not set in TBRMMParameters. The budget constraint will be "
            "ignored by the original library's search algorithm."
        )
      # for 2 cell budget.value is either a float or a list of 1 float
      treatment_budget_value = (
          budget.value[0] if isinstance(budget.value, list) else budget.value
      )

      if (budget.budget_type ==
          geoflex.experiment_design.ExperimentBudgetType.DAILY_BUDGET):
        # calculate total budget change by multiplying daily budget by runtime
        total_budget_change = (
            treatment_budget_value *
            geoflex_design_obj.runtime_weeks * 7)
        original_params_kwargs["budget_range"] = (
            0.0,
            abs(total_budget_change),
        )
        logger.info(
            "Mapped GeoFleX budget type '%s' with value %s (daily) to original"
            " library budget_range (0.0, %s) (total over %s runtime weeks).",
            budget.budget_type.value,
            treatment_budget_value,
            abs(total_budget_change),
            geoflex_design_obj.runtime_weeks,
        )
      elif (budget.budget_type ==
            geoflex.experiment_design.ExperimentBudgetType.TOTAL_BUDGET):
        # assume total_budget is total allowable change
        original_params_kwargs["budget_range"] = (
            0.0,
            abs(treatment_budget_value),
        )
        logger.info(
            "Mapped GeoFleX budget type '%s' with value %s (total) to original"
            " library budget_range (0.0, %s).",
            budget.budget_type.value,
            treatment_budget_value,
            abs(treatment_budget_value),
        )
      elif (
          budget.budget_type
          == geoflex.experiment_design.ExperimentBudgetType.PERCENTAGE_CHANGE
      ):
        logger.info(
            "GeoFleX budget type '%s' cannot be mapped to original library's"
            " absolute budget_range without BAU spend. budget_range will not"
            " be passed.",
            budget.budget_type.value,
        )
    else:
      logger.info(
          "No GeoFleX experiment budget specified. budget_range will not be"
          " passed to original library."
      )

    logger.debug(
        "Final kwargs for original TBRMMDesignParameters: %s",
        original_params_kwargs
        )
    try:
      return original_tbrmmdesignparameters.TBRMMDesignParameters(
          **original_params_kwargs
      )
    except ValueError as e:
      logger.error(
          "Error instantiating original TBRMMDesignParameters"
          " with kwargs %s: %s",
          original_params_kwargs, e, exc_info=True
          )
      raise ValueError(
          "Failed to create valid TBRMMDesignParameters for original library."
          ) from e

  def _adapt_geoflex_data_to_original_tbr_input(
      self,
      runtime_data: GeoPerformanceDataset,
      assignment: GeoAssignment,
      metric_column_names: list[str],
      pretest_weeks: int,
      experiment_start_date: pd.Timestamp,
      experiment_end_date: pd.Timestamp,
      date_col_name: str,
      geo_col_name: str,
  ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Converts GeoFleX GeoPerformanceDataset for the original library.

    The original google/matched_markets library expects a specific input
    format to be passed to the TBR fit function. This version adapts data
    for multiple metrics simultaneously.

    Args:
      runtime_data: GeoPerformanceDataset object containing the data.
      assignment: GeoAssignment object containing the geo assignment.
      metric_column_names: The column names of the metrics to use in the TBR
        analysis.
      pretest_weeks: The number of weeks in the pretest period.
      experiment_start_date: The start date of the experiment.
      experiment_end_date: The end date of the experiment.
      date_col_name: The column name of the date column.
      geo_col_name: The column name of the geo
    Returns:
      A tuple of (final_df_for_tbr, tbr_fit_kwargs), where:
      - final_df_for_tbr: A DataFrame containing the data for TBR analysis
        with all specified metric columns.
      - tbr_fit_kwargs: A dictionary of keyword arguments for the TBR fit
        function.
    """
    logger.debug(
        "Adapting GeoFleX data to original TBR format for metrics '%s'.",
        metric_column_names
        )

    parsed_data = runtime_data.parsed_data
    control_geos = assignment.control
    treatment_geos = assignment.treatment[0]
    start_dt = pd.to_datetime(experiment_start_date)
    pretest_start_dt = start_dt - pd.Timedelta(weeks=pretest_weeks)
    min_data_date = parsed_data[date_col_name].min()

    if pretest_start_dt < min_data_date:
      pretest_start_dt = min_data_date

    analysis_data_slice = parsed_data[
        (parsed_data[date_col_name] >= pretest_start_dt)
        & (parsed_data[date_col_name] < experiment_end_date)
    ].copy()

    # Aggregate data for control and treatment groups for all metrics
    control_ts_df = self._aggregate_data_for_group(
        control_geos,
        metric_column_names,
        analysis_data_slice,
        geo_col_name,
        date_col_name,
        runtime_data.data_frequency_days)

    treatment_ts_df = self._aggregate_data_for_group(
        treatment_geos,
        metric_column_names,
        analysis_data_slice,
        geo_col_name,
        date_col_name,
        runtime_data.data_frequency_days)

    sem = {
        "key_date": "date",
        "key_group": "group",
        "key_period": "period",
        "group_control": 1,
        "group_treatment": 2,
        "period_pre": 0,
        "period_test": 1
        }

    # Prepare DataFrames for concatenation, adding group and period columns
    control_df = control_ts_df.reset_index()
    control_df.rename(columns={date_col_name: sem["key_date"]}, inplace=True)
    control_df[sem["key_group"]] = sem["group_control"]

    treatment_df = treatment_ts_df.reset_index()
    treatment_df.rename(columns={date_col_name: sem["key_date"]}, inplace=True)
    treatment_df[sem["key_group"]] = sem["group_treatment"]

    combined_df = pd.concat([control_df, treatment_df], ignore_index=True)
    combined_df[sem["key_period"]] = np.where(
        combined_df[sem["key_date"]] < start_dt,
        sem["period_pre"],
        sem["period_test"]
        )

    # Select only the required columns for the final DataFrame
    required_cols = [
        sem["key_date"], sem["key_group"], sem["key_period"]
        ] + metric_column_names
    final_df_for_tbr = combined_df[required_cols]

    tbr_fit_kwargs = {
        key: sem[key] for key in
        ["key_date", "key_group", "key_period", "group_control",
         "group_treatment", "period_pre", "period_test"]
        }
    return final_df_for_tbr, tbr_fit_kwargs

  def _adapt_original_tbr_summary_to_geoflex(
      self,
      original_summary_row: pd.Series,
      control_total_runtime: float,
      alternative_hypothesis: str,
      alpha: float,
      metric_name: str
  ) -> dict[str, Any]:
    """Converts the original TBR summary to the GeoFleX format.

    The original google/matched_markets library returns a summary of the TBR
    analysis as a dictionary. The GeoFleX analysis expects a dictionary of
    summary statistics in a different format so conversion is required.

    Args:
      original_summary_row: The original TBR summary row.
      control_total_runtime: The total runtime of the control group.
      alternative_hypothesis: The alternative hypothesis.
      alpha: The significance level.
      metric_name: The name of the metric.

    Returns:
      A dictionary of the adapted summary.
    """
    logger.debug(
        "Adapting original TBR summary to GeoFleX format for metric '%s'.",
        metric_name
        )

    point_estimate = original_summary_row.get("estimate", pd.NA)
    lower_bound = original_summary_row.get("lower", pd.NA)
    upper_bound = original_summary_row.get("upper", pd.NA)
    standard_error = original_summary_row.get("scale", pd.NA)
    p_value = pd.NA

    if (pd.notna(point_estimate) and
        pd.notna(standard_error) and
        standard_error > 1e-9):
      t_stat = point_estimate / standard_error
      df_resid_original = original_summary_row.get("df_resid")

      if pd.notna(df_resid_original) and df_resid_original > 0:
        df_resid_approx = df_resid_original
      else:
        df_resid_approx = 1000  # Default degrees of freedom
        logger.info(
            "Original TBR summary for metric '%s' did not provide a valid"
            " 'df_resid' (got %s). Using approximate df_resid=%s for p-value"
            " calculation.",
            metric_name, df_resid_original, df_resid_approx
        )

      if alternative_hypothesis == "two-sided":
        p_value = 2 * stats.t.sf(np.abs(t_stat), df=df_resid_approx)

      elif alternative_hypothesis == "greater":
        p_value = stats.t.sf(t_stat, df=df_resid_approx)

      elif alternative_hypothesis == "less":
        p_value = stats.t.cdf(t_stat, df=df_resid_approx)
      p_value = min(p_value, 1.0) if pd.notna(p_value) else pd.NA

    is_significant = pd.notna(p_value) and p_value < alpha
    pe_rel, lb_rel, ub_rel = self._get_relative_metrics(
        point_estimate,
        lower_bound,
        upper_bound,
        control_total_runtime,
        alternative_hypothesis
        )
    return {
        "point_estimate": point_estimate,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "point_estimate_relative": pe_rel,
        "lower_bound_relative": lb_rel,
        "upper_bound_relative": ub_rel,
        "p_value": p_value,
        "is_significant": bool(is_significant) if (
            pd.notna(is_significant)) else pd.NA,
        "_standard_error": standard_error
        }

  def _get_relative_metrics(
      self,
      point_estimate: float,
      lower_bound: float,
      upper_bound: float,
      control_total_runtime: float,
      alternative_hypothesis: str
  ) -> tuple[Any, Any, Any]:
    """Calculates relative metrics based on the control total runtime.

    Args:
        point_estimate: The point estimate.
        lower_bound: The lower bound.
        upper_bound: The upper bound.
        control_total_runtime: The total runtime of the control group.
        alternative_hypothesis: The alternative hypothesis (unused in this
          specific calculation but often part of such method signatures).

    Returns:
        A tuple containing the relative point estimate, lower bound, and upper
        bound. Returns (pd.NA, pd.NA, pd.NA) if control_total_runtime is zero
        or NA.
    """
    if pd.isna(control_total_runtime) or control_total_runtime == 0:
      logger.warning(
          "Cannot calculate relative metrics with zero or NA control total"
          " runtime. Returning NA for relative metrics."
      )
      return pd.NA, pd.NA, pd.NA

    pe_rel = (
        point_estimate / control_total_runtime
        if pd.notna(point_estimate)
        else pd.NA
    )
    lb_rel = (
        lower_bound / control_total_runtime
        if pd.notna(lower_bound)
        else pd.NA
    )
    ub_rel = (
        upper_bound / control_total_runtime
        if pd.notna(upper_bound)
        else pd.NA
    )
    return pe_rel, lb_rel, ub_rel

  def _get_control_total_for_relative(
      self,
      adapted_long_format_df,
      metric_column_name,
      semantic_kwargs
  ) -> float:
    """Calculates total runtime of the control group for relative metrics."""
    control_df = adapted_long_format_df[
        (adapted_long_format_df[semantic_kwargs["key_group"]] ==
         semantic_kwargs["group_control"]) & (
             adapted_long_format_df[semantic_kwargs["key_period"]] ==
             semantic_kwargs["period_test"])]

    return control_df[metric_column_name].sum()

  def _get_empty_analysis_result_row(
      self,
      cell_number: int,
      metric: Metric,
      design: ExperimentDesign
  ) -> dict[str, Any]:
    """Adds an empty analysis result row for a given cell, metric, and design.

    Required for the original matched_markets library output format.

    Args:
      cell_number: The cell number.
      metric: The metric to analyze.
      design: The experiment design.
    Returns:
      Empty analysis result row.
    """
    return {
        "cell": cell_number,
        "metric": metric.name,
        "is_primary_metric": metric == design.primary_metric,
        "point_estimate": pd.NA,
        "lower_bound": pd.NA,
        "upper_bound": pd.NA,
        "point_estimate_relative": pd.NA,
        "lower_bound_relative": pd.NA,
        "upper_bound_relative": pd.NA,
        "p_value": pd.NA,
        "is_significant": pd.NA,
        "_standard_error": pd.NA
        }

  def _aggregate_data_for_group(
      self,
      geos: set[str],
      metric_cols: list[str],
      data_df: pd.DataFrame,
      geo_col: str,
      date_col: str,
      data_frequency_days: int,
  ) -> pd.DataFrame:
    """Aggregates data for a given group of geos across multiple metrics.

    Args:
      geos: The set of geos to aggregate.
      metric_cols: The column names of the metrics to aggregate.
      data_df: The DataFrame containing the data.
      geo_col: The column name of the geo column.
      date_col: The column name of the date column.
      data_frequency_days: The number of days per data point in the
        historical data (e.g., 1 for daily, 7 for weekly).

    Returns:
      A DataFrame containing the aggregated data for all specified metrics.
    """
    if data_df.empty or not metric_cols:
      # Return an empty DataFrame with columns for the specified metrics
      return pd.DataFrame(
          columns=metric_cols,
          index=pd.to_datetime([]).rename(date_col))

    # ensure there is datetime index or a date column that can be set as index
    if not isinstance(data_df.index, pd.DatetimeIndex):
      if (date_col in data_df.columns and
          pd.api.types.is_datetime64_any_dtype(data_df[date_col])):
        current_data_df = data_df.set_index(date_col)
      else:  # attempt conversion
        current_data_df = data_df.copy()
        try:
          current_data_df[date_col] = pd.to_datetime(current_data_df[date_col])
          current_data_df = current_data_df.set_index(date_col)
        except (ValueError, TypeError) as e:
          raise ValueError(
              f"Cannot set datetime index using column '{date_col}': {e}"
              ) from e
    else:
      current_data_df = data_df

    min_date, max_date = (
        current_data_df.index.min(), current_data_df.index.max()
    )
    # attempt to get frequency from GeoPerformanceDataset attribute
    # default to 'D' if no other info is available
    if data_frequency_days == 7:
      freq_str = "W"  # assuming weekly data aligns with pandas' week frequency
    elif data_frequency_days == 1:
      freq_str = "D"
    # fallback if other frequency is not supported by original library
    else:
      freq_str = f"{data_frequency_days}D"
      logger.warning(
          "Using custom frequency string %s for date_range.", freq_str)

    date_index_for_reindex = pd.date_range(
        start=min_date, end=max_date, freq=freq_str, name=date_col
        )
    if not geos:
      # Return a DataFrame of zeros with the correct index and columns
      return pd.DataFrame(
          0.0,
          index=date_index_for_reindex,
          columns=metric_cols,
          dtype=float)

    group_data = current_data_df[current_data_df[geo_col].isin(geos)]
    if group_data.empty:
      # Return a DataFrame of zeros with the correct index and columns
      return pd.DataFrame(
          0.0,
          index=date_index_for_reindex,
          columns=metric_cols,
          dtype=float)

    # Aggregate multiple metric columns
    agg_df = group_data.groupby(group_data.index)[metric_cols].sum(min_count=1)
    return agg_df.reindex(date_index_for_reindex, fill_value=0.0)
