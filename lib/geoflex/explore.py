# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A module for exploring experiment designs to find the optimal one."""

import datetime as dt
import functools
import logging
from typing import Any
import warnings
import geoflex.data
import geoflex.evaluation
import geoflex.experiment_design
import geoflex.exploration_spec
import geoflex.methodology
import numpy as np
import optuna as op
import pandas as pd
import pydantic
import tqdm.auto as tqdm

ExperimentDesignExplorationSpec = (
    geoflex.exploration_spec.ExperimentDesignExplorationSpec
)

GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesignEvaluator = geoflex.evaluation.ExperimentDesignEvaluator
ExperimentDesign = geoflex.experiment_design.ExperimentDesign
EffectScope = geoflex.experiment_design.EffectScope
assign_geos = geoflex.methodology.assign_geos
design_is_eligible_for_data = geoflex.methodology.design_is_eligible_for_data
get_methodology = geoflex.methodology.get_methodology

logger = logging.getLogger(__name__)


class MaxTrialsCallback:
  """Used to stop an Optuna study after a certain number of trials.

  With the way we are using Optuna, some trials can be invalid and will return
  inf for the first value. We want to run Optuna for a certain number of
  valid trials, so we use this callback to stop the study after we reach that
  number.
  """

  def __init__(self, max_trials: int, with_progress_bar: bool = False):
    """Initializes the callback.

    Args:
      max_trials: The maximum number of valid trials to run.
      with_progress_bar: Whether to show a progress bar. Defaults to False. It's
        recommended to set this to True only if you are not also printing info
        logs to the console.
    """
    self.max_trials = max_trials
    self.with_progress_bar = with_progress_bar

    if with_progress_bar:
      self.pbar = tqdm.tqdm(total=max_trials, desc="GeoFleX Exploration")

  @staticmethod
  def get_n_completed_trials(study: op.Study) -> int:
    """Returns the number of completed trials in the study."""
    return len([
        t
        for t in study.trials
        if t.values is not None and not np.isinf(t.values[0])
    ])

  def __call__(self, study: op.Study, trial: op.Trial) -> None:
    n_non_inf_trials = self.get_n_completed_trials(study)

    if (
        self.with_progress_bar
        and trial.values is not None
        and not np.isinf(trial.values[0])
    ):
      self.pbar.update(1)

    if n_non_inf_trials == self.max_trials:
      logger.info("Stopping study after %s trials.", n_non_inf_trials)
      study.stop()
      if self.with_progress_bar:
        self.pbar.close()
    elif n_non_inf_trials > self.max_trials:
      logger.info(
          "Stopping study after %s trials. Overshot the max trials (%s) by %s,"
          " likely because of parallelisation.",
          n_non_inf_trials,
          self.max_trials,
          n_non_inf_trials - self.max_trials,
      )
      study.stop()
      if self.with_progress_bar:
        self.pbar.close()


class ExperimentDesignExplorer(pydantic.BaseModel):
  """The class for exploring experiment designs.

  Attributes:
    historical_data: The historical data to design the experiment for.
    explore_spec: The experiment design exploration spec.
    bootstrapper_seasons_per_block: The number of seasons per block to use for
      the bootstrapper.
    bootstrapper_model_type: The model type to use for the bootstrapper. Either
      "additive" or "multiplicative".
    bootstrapper_seasonality: The seasonality to use for the bootstrapper.
    bootstrapper_sampling_type: The sampling type to use for the bootstrapper.
      Either "permutation" or "random".
    bootstrapper_stl_params: The parameters for the STL model. Defaults to 0 for
      seasonal degree, 0 for trend degree, 0 for low pass degree, and False for
      robust.
    validation_check_threhold: The threhold to use for the validation check.
      Defaults to 0.001, which is a 99.9% confidence level. Typically this does
      not need to be changed.
    explored_designs: The explored designs. This is a dictionary of design_id to
      ExperimentDesign. Every time a new design is explored, the design will be
      added to this dictionary.
    study: The Optuna study used to explore the experiment designs.
    pareto_front_design_ids: The design ids that are on the Pareto front (most
      optimal).
  """

  historical_data: GeoPerformanceDataset
  explore_spec: ExperimentDesignExplorationSpec

  bootstrapper_seasons_per_block: int = 4
  bootstrapper_model_type: str = "multiplicative"
  bootstrapper_seasonality: int = 7
  bootstrapper_sampling_type: str = "permutation"
  bootstrapper_stl_params: dict[str, Any] | None = None

  validation_check_threhold: float = 0.001

  explored_designs: dict[str, ExperimentDesign] = {}
  study: op.Study | None = None
  pareto_front_design_ids: list[str] = []

  model_config = pydantic.ConfigDict(
      extra="forbid", arbitrary_types_allowed=True
  )

  @pydantic.field_serializer("study")
  def serialize_study_as_none(
      self, study: op.Study | None, info: pydantic.SerializationInfo
  ) -> None:
    """Ensures that study is always None in the JSON output.

    This is because Optuna does not support JSON serialization.

    Args:
      study: The Optuna study.
      info: The serialization info.

    Returns:
      None
    """
    del info, study  # unused
    return None

  def clear_designs(self) -> None:
    """Clears the explored designs."""
    self.explored_designs = {}

  @functools.cached_property
  def _experiment_design_evaluator(self) -> ExperimentDesignEvaluator:
    """Returns the experiment design evaluator."""
    return ExperimentDesignEvaluator(
        historical_data=self.historical_data,
        bootstrapper_seasons_per_block=self.bootstrapper_seasons_per_block,
        bootstrapper_model_type=self.bootstrapper_model_type,
        bootstrapper_seasonality=self.bootstrapper_seasonality,
        bootstrapper_sampling_type=self.bootstrapper_sampling_type,
        bootstrapper_stl_params=self.bootstrapper_stl_params,
        validation_check_threhold=self.validation_check_threhold,
    )

  @functools.cached_property
  def exp_start_date(self) -> dt.date:
    """Returns the synthetic experiment start date for the simulations."""
    max_runtime_weeks = max(self.explore_spec.runtime_weeks_candidates)
    final_date = self.historical_data.parsed_data[
        self.historical_data.date_column
    ].max()
    return final_date - dt.timedelta(weeks=max_runtime_weeks)

  def _suggest_methodology_parameters(
      self,
      trial: op.Trial,
      methodology_name: str,
  ) -> dict[str, Any]:
    """Suggests methodology parameters for the given methodology and trial.

    This will suggest values for the methodology parameters that are specific
    to the methodology, based on the design and the trial. The suggested
    values will be added to the design. It will check the explore spec for
    any parameter candidates and will use those if they are specified, otherwise
    it will use the default parameter candidates that are specific to the
    methodology.

    Args:
      trial: The trial to use to suggest the parameters.
      methodology_name: The methodology to suggest parameters for.

    Returns:
      A dictionary of parameter names and values.
    """
    methodology = get_methodology(methodology_name)

    restricted_parameter_candidates = (
        self.explore_spec.methodology_parameter_candidates.get(
            methodology_name, {}
        )
    )
    methodology_parameters = {}
    for (
        parameter_name,
        default_parameter_candidates,
    ) in methodology.default_methodology_parameter_candidates.items():
      parameter_candidates = restricted_parameter_candidates.get(
          parameter_name, default_parameter_candidates
      )
      parameter_candidates_ids = list(range(len(parameter_candidates)))
      parameter_id_name = f"{methodology_name}_{parameter_name}_id"

      parameter_id = trial.suggest_categorical(
          parameter_id_name, parameter_candidates_ids
      )
      methodology_parameters[parameter_name] = parameter_candidates[
          parameter_id
      ]

    return methodology_parameters

  def _suggest_experiment_design(self, trial: op.Trial) -> ExperimentDesign:
    """Suggests an experiment design for the given trial.

    It will suggest experiment designs based on the experiment design spec.

    Args:
      trial: The Optuna trial to use to suggest the experiment design.

    Returns:
      The suggested experiment design.
    """
    runtime_weeks = trial.suggest_categorical(
        "runtime_weeks",
        self.explore_spec.runtime_weeks_candidates,
    )
    methodology_name = trial.suggest_categorical(
        "methodology", list(self.explore_spec.eligible_methodologies)
    )
    geo_eligibility_id = trial.suggest_categorical(
        "geo_eligibility_id",
        list(range(len(self.explore_spec.geo_eligibility_candidates))),
    )
    cell_volume_constraint_id = trial.suggest_categorical(
        "cell_volume_constraint_id",
        list(range(len(self.explore_spec.cell_volume_constraint_candidates))),
    )
    random_seed = trial.suggest_categorical(
        "random_seed", self.explore_spec.random_seeds
    )
    experiment_budget_id = trial.suggest_categorical(
        "experiment_budget_id",
        list(range(len(self.explore_spec.experiment_budget_candidates))),
    )

    geo_eligibility = self.explore_spec.geo_eligibility_candidates[
        geo_eligibility_id
    ]
    cell_volume_constraint = (
        self.explore_spec.cell_volume_constraint_candidates[
            cell_volume_constraint_id
        ]
    )
    experiment_budget = self.explore_spec.experiment_budget_candidates[
        experiment_budget_id
    ]

    methodology_parameters = self._suggest_methodology_parameters(
        trial, methodology_name
    )

    design = ExperimentDesign(
        primary_metric=self.explore_spec.primary_metric,
        experiment_budget=experiment_budget,
        secondary_metrics=self.explore_spec.secondary_metrics,
        methodology=methodology_name,
        methodology_parameters=methodology_parameters,
        runtime_weeks=runtime_weeks,
        n_cells=self.explore_spec.n_cells,
        alpha=self.explore_spec.alpha,
        alternative_hypothesis=self.explore_spec.alternative_hypothesis,
        geo_eligibility=geo_eligibility,
        cell_volume_constraint=cell_volume_constraint,
        random_seed=random_seed,
        effect_scope=self.explore_spec.effect_scope,
    )

    return design

  def _exploration_objective(
      self,
      trial: op.Trial,
      aa_simulations_per_trial: int | None,
      ab_simulations_per_trial: int | None,
  ) -> tuple[float, float]:
    """Suggests a new experiment design, and evaluates it.

    This is the objective function for optuna. It will suggest a new experiment
    design based on the design spec, simulate the experiment, and evaluate the
    results. The evaluation will determine if the design is eligible for the
    methodology, and if the primary metric meets the validation checks.

    If the design is not eligible for the methodology, we will return inf for
    the standard error, and -1.0 for the representativeness score. This will
    ensure that this design is never selected.

    Args:
      trial: The Optuna trial to use to suggest the experiment design.
      aa_simulations_per_trial: The number of simulations to run per trial. If
        None, the number of simulations will be automatically determined to
        ensure statistically rigorous evaluations.
      ab_simulations_per_trial: The number of simulations to run per trial. If
        None, the number of simulations will be automatically determined to
        ensure statistically rigorous evaluations.

    Returns:
      A tuple of the standard error of the primary metric, and the
      representativeness score. If the relative standard error is not None for
      the primary metric, then this will be returned, otherwise the absolute
      standard error will be returned. If the design is not eligible for the
      methodology, or if the design fails the checks and
      ignore_designs_with_failing_checks is True, then inf will be returned for
      the standard error, and -1.0 for the representativeness score.
    """
    # Suggest the experiment design based on the design spec.
    design = self._suggest_experiment_design(trial)
    trial.set_user_attr("design_id", design.design_id)

    # Simulate the experiment with the suggested design.
    evaluation_results = self._experiment_design_evaluator.evaluate_design(
        design=design,
        exp_start_date=self.exp_start_date,
        n_aa_simulations=aa_simulations_per_trial,
        n_ab_simulations=ab_simulations_per_trial,
        overwrite_mode="overwrite",
    )

    # If the design is not eligible for the methdology, the results will be
    # None. In this case we return inf for the standard error, and -1.0 for the
    # representativeness score. This will ensure that this design is never
    # selected.
    if not evaluation_results.is_compatible_design:
      logger.info(
          "Design %s (trial %s) is not eligible for methodology %s",
          design.design_id,
          trial.number,
          design.methodology,
      )
      return np.inf, -1.0

    # Get the standard error and representativeness score for the design.
    primary_metric_standard_error = (
        evaluation_results.primary_metric_results.standard_error_relative_effect
        or evaluation_results.primary_metric_results.standard_error_absolute_effect
    )
    representativeness_score = evaluation_results.representativeness_score

    # If the design fails any checks, then log the failing checks.
    if not evaluation_results.primary_metric_results.all_checks_pass:
      failed_checks = ", ".join(
          list(set(evaluation_results.primary_metric_results.failing_checks))
      )
      logger.info(
          "Design %s (trial %s) does not meet the validation checks for the"
          " primary metric, the following checks failed: %s",
          design.design_id,
          trial.number,
          failed_checks,
      )

    # Record the design as explored, regardless of whether the checks pass.
    self.explored_designs[design.design_id] = design

    # Finally we return the standard error of the primary metric, and the
    # representativeness score. These will be used by optuna to optimise the
    # objective. We will perform multi-objective optimisation to find the
    # smallest standard error and highest representativeness score.
    return primary_metric_standard_error, representativeness_score

  def _design_counting_objective(
      self,
      trial: op.Trial,
  ) -> float:
    """A dummy objective function used to count the number of eligible designs.

    This doesn't actually evaluate the design, it just logs the number of
    eligible designs for each methodology. This is used to count the number
    of eligible designs for each methodology.

    Args:
      trial: The Optuna trial to use to suggest the experiment design.

    Returns:
      The score is 1 if the design is eligible for the methodology, 0 otherwise.
    """
    # Suggest the experiment design based on the design spec.
    design = self._suggest_experiment_design(trial)
    if design_is_eligible_for_data(design, self.historical_data):
      trial.set_user_attr("methodology", design.methodology)
      return 1.0
    else:
      return 0.0

  def count_all_eligible_designs(self) -> dict[str, int]:
    """Returns the number of eligible experiment designs per methodology.

    Note: this does not look at the designs that have already been explored,
    it looks at all possible designs regardless of whether they have already
    been explored.

    If there are more than 10k eligible designs, it will only count up to 10k.
    """
    # Get current global logging level
    previous_global_logging_level = logging.getLogger().getEffectiveLevel()
    # Set global logging level to ERROR
    logging.getLogger().setLevel(logging.ERROR)

    try:
      previous_verbosity = op.logging.get_verbosity()
      op.logging.set_verbosity(logging.CRITICAL)  # Disable logging.
      try:
        with warnings.catch_warnings():
          # Hide the experiment warning about the BruteForceSampler being
          # experimental.
          warnings.filterwarnings(
              "ignore", message="BruteForceSampler is experimental"
          )

          counting_study = op.create_study(
              sampler=op.samplers.BruteForceSampler(), directions=["maximize"]
          )

        counting_study.optimize(
            self._design_counting_objective,
            n_trials=10_000,
            n_jobs=-1,
        )

        all_results = counting_study.trials_dataframe()

        counts = (
            all_results.drop_duplicates()
            .groupby("user_attrs_methodology")["value"]
            .sum()
            .astype(int)
            .to_dict()
        )
      finally:
        op.logging.set_verbosity(previous_verbosity)  # Restore logging.
      return counts
    finally:
      # Restore global logging level
      logging.getLogger().setLevel(previous_global_logging_level)

  def explore(
      self,
      max_trials: int,
      aa_simulations_per_trial: int | None = None,
      ab_simulations_per_trial: int | None = None,
      n_jobs: int = 1,
      seed: int = 0,
      warm_start: bool = True,
      with_progress_bar: bool = False,
  ) -> None:
    """Explores experiment designs.

    Args:
      max_trials: The maximum number of trials to explore.
      aa_simulations_per_trial: The number of A/A simulations to run per trial.
        If None, the number of A/A simulations will be automatically determined
        to ensure statistically rigorous evaluations.
      ab_simulations_per_trial: The number of A/B simulations to run per trial.
        If None, the number of A/B simulations will be automatically determined
        to ensure statistically rigorous evaluations.
      n_jobs: The number of parallel jobs to use for the optimization. Defaults
        to -1, which means all available cores will be used.
      seed: The seed to use for the optimization.
      warm_start: Whether to warm start the optimization.
      with_progress_bar: Whether to show a progress bar while exploring the
        designs. Defaults to False. It's recommended to set this to True only if
        you are not also printing info logs to the console.
    """

    # Call the bootstrapper to make it fit to the data.
    _ = self._experiment_design_evaluator.bootstrapper

    objective = lambda trial: self._exploration_objective(
        trial,
        aa_simulations_per_trial=aa_simulations_per_trial,
        ab_simulations_per_trial=ab_simulations_per_trial,
    )

    if warm_start and self.study is None and self.explored_designs:
      error_message = (
          "Warm start is not supported when there are existing experiment"
          " designs but no existing study. This usually happens if you are"
          " loading the experiment from json and then exploring with warm"
          " start, as the study cannot be serialized."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    if not warm_start:
      logger.info(
          "Warm start is disabled, clearing existing designs and study."
      )
      self.clear_designs()

    has_existing_study = self.study is not None
    has_existing_designs = bool(self.explored_designs)

    if not has_existing_study and has_existing_designs and warm_start:
      error_message = (
          "Warm start is not supported when there are existing experiment"
          " designs but no existing study. This usually happens if you are"
          " loading the experiment from json and then exploring with warm"
          " start, as the study cannot be serialized."
      )
      logger.error(error_message)
      raise ValueError(error_message)
    elif has_existing_study and warm_start:
      logger.info("Continuing existing study.")
    else:
      logger.info("Creating new study.")
      with warnings.catch_warnings():
        # Hide the experiment warning about the BruteForceSampler being
        # experimental.
        warnings.filterwarnings(
            "ignore", message="BruteForceSampler is experimental"
        )
        self.study = op.create_study(
            sampler=op.samplers.BruteForceSampler(seed=seed),
            directions=[
                "minimize",
                "maximize",
            ],  # Minimise standard error, maximise representativeness
        )

    existing_trials = MaxTrialsCallback.get_n_completed_trials(self.study)
    target_trials = existing_trials + max_trials

    if existing_trials:
      logger.info(
          "Study already has %s trials, exploring for max %s trials, so there"
          " will be max %s trials in total after exploration is complete.",
          existing_trials,
          max_trials,
          target_trials,
      )
    else:
      logger.info("No existing trials, exploring for max %s trials", max_trials)

    self.study.optimize(
        objective,
        n_trials=max_trials * 10,
        callbacks=[MaxTrialsCallback(target_trials, with_progress_bar)],
        n_jobs=n_jobs,
        catch=Exception,  # Catch all exceptions, never stop the study.
    )

    trials_dataframe = self.study.trials_dataframe()
    if "user_attrs_design_id" in trials_dataframe.columns:
      best_trial_numbers = [trial.number for trial in self.study.best_trials]
      self.pareto_front_design_ids = (
          self.study.trials_dataframe()
          .loc[best_trial_numbers, "user_attrs_design_id"]
          .values.tolist()
      )
    else:
      logger.warning(
          "user_attrs_design_id not found in trials dataframe. This usually "
          "happens because there are no valid trials."
      )
      self.pareto_front_design_ids = []

  def get_designs(
      self,
      top_n: int | None = None,
      pareto_front_only: bool = False,
      sort_failing_designs_last: bool = False,
  ) -> list[ExperimentDesign]:
    """Returns the explored experiment designs.

    Args:
      top_n: The number of top designs to return, ranked by the MDE of the
        primary metric. If None, all designs will be returned.
      pareto_front_only: Whether to only return the pareto front designs. These
        are the designs that have the smallest MDE for the primary metric, and
        the highest representativeness score.
      sort_failing_designs_last: Whether to sort the failing designs last.

    Returns:
      The explored experiment designs.
    """
    if pareto_front_only:
      all_designs = [
          self.explored_designs[design_id]
          for design_id in self.pareto_front_design_ids
      ]
    else:
      all_designs = list(self.explored_designs.values())

    def _get_sort_key(design: ExperimentDesign) -> float:
      """Sort by the relative standard error if exists, or else absolute."""
      if not design.evaluation_results:
        return np.inf

      relative_standard_error = (
          design.evaluation_results.primary_metric_results.standard_error_relative_effect
      )
      absolute_standard_error = (
          design.evaluation_results.primary_metric_results.standard_error_absolute_effect
      )

      standard_error = relative_standard_error or absolute_standard_error
      if sort_failing_designs_last:
        if not design.evaluation_results.get_summary_dict()["all_checks_pass"]:
          # If any checks fail, worst case
          failing_index = 2
        elif not design.evaluation_results.sufficient_simulations:
          # If they don't fail but don't have enough simulations, that's better
          failing_index = 1
        else:
          # If they pass and have enough simulations, that's best
          failing_index = 0

        return (
            failing_index,
            standard_error,
        )
      else:
        return standard_error

    all_designs.sort(
        key=_get_sort_key,
        reverse=False,
    )

    if top_n is not None:
      all_designs = all_designs[:top_n]

    return all_designs

  def get_design_summaries(
      self,
      top_n: int | None = None,
      pareto_front_only: bool = False,
      sort_failing_designs_last: bool = False,
      target_power: float = 0.8,
      target_primary_metric_mde: float | None = None,
      use_relative_effects_where_possible: bool = True,
      drop_constant_columns: bool = False,
      shorten_geo_assignments: bool = True,
      style_output: bool = False,
  ) -> pd.DataFrame | pd.io.formats.style.Styler:
    """Returns the summaries of the explored experiment designs.

    If the experiment is a multi-cell experiment, then the performance metrics
    (the MDEs, standard errors, coverages and failing checks) are taken to be
    the worst case scenario for each cell. So the standard error is the maximum
    standard error across all cells, and the coverage is the minimum coverage
    across all cells, etc. This is to ensure that the best design selected is
    one that works across all cells in the design.

    Args:
      top_n: The number of top designs to return, ranked by the MDE of the
        primary metric. If None, all designs will be returned.
      pareto_front_only: Whether to only include the pareto front of the
        experiment designs. The pareto front is the set of designs that have the
        best performance on both representativeness and performance metrics.
      sort_failing_designs_last: Whether to sort the failing designs last.
      target_power: The target power to use for the MDE calculations. The MDE
        will be calculated for every metric specified in the design spec.
      target_primary_metric_mde: The target MDE for the primary metric. If this
        is set, then the power will be calculated for the primary metric at this
        MDE.
      use_relative_effects_where_possible: Whether to use the relative effects
        where possible. For each metric, if the metric is a cost-per-metric or
        metric-per-cost metric, we will always use the absolute effect.
        Otherwise, if this is set to true we will use relative effects.
      drop_constant_columns: Whether to drop the constant columns from the
        dataframe. These are the columns that are the same for all designs. It
        can make it easier to compare the designs side-by-side by looking at
        only the differences.
      shorten_geo_assignments: Shorten the columns containing the geo
        assignments to make them more readable.
      style_output: Whether to return the dataframe as a Styler object with
        formatting.

    Returns:
      A dataframe containing the experiment design summaries.
    """
    all_metrics = [
        self.explore_spec.primary_metric
    ] + self.explore_spec.secondary_metrics
    cost_per_metric_metrics = [
        metric.name for metric in all_metrics if metric.cost_per_metric
    ]
    include_representativeness_score = (
        self.explore_spec.effect_scope == EffectScope.ALL_GEOS
    )

    designs = self.get_designs(
        top_n,
        pareto_front_only=pareto_front_only,
        sort_failing_designs_last=sort_failing_designs_last,
    )
    return geoflex.experiment_design.compare_designs(
        designs=designs,
        target_power=target_power,
        target_primary_metric_mde=target_primary_metric_mde,
        use_relative_effects_where_possible=use_relative_effects_where_possible,
        shorten_geo_assignments=shorten_geo_assignments,
        drop_constant_columns=drop_constant_columns,
        style_output=style_output,
        cost_per_metric_metrics=cost_per_metric_metrics,
        include_representativeness_score=include_representativeness_score,
    )

  def get_design_by_id(self, design_id: str) -> ExperimentDesign | None:
    """Returns the experiment design with the given ID, or None if not found.

    Args:
      design_id: The ID of the experiment design to get.

    Returns:
      The experiment design with the given ID, or None if not found.
    """
    return self.explored_designs.get(design_id)

  def extend_top_n_designs(
      self,
      top_n: int,
      n_aa_simulations: int | None = None,
      n_ab_simulations: int | None = None,
      with_progress_bar: bool = False,
  ) -> None:
    """Extends the top n designs with the given number of A/A and A/B simulations.

    This can be useful if you first run a small number of simulations to get
    the designs, and then you want to run more simulations on the designs that
    you like.

    Selects the top n designs ranked by the MDE of the primary metric,
    prioritizing those that have not failed any checks.

    Args:
      top_n: The number of top designs to extend.
      n_aa_simulations: The number of extra A/A simulations to run. If None, the
        number of A/A simulations will be automatically determined to ensure
        statistically rigorous evaluations.
      n_ab_simulations: The number of extra A/B simulations to run. If None, the
        number of A/B simulations will be automatically determined to ensure
        statistically rigorous evaluations.
      with_progress_bar: Whether to show a progress bar while extending the
        designs. Defaults to False. It's recommended to set this to True only if
        you are not also printing info logs to the console.
    """
    designs = self.get_designs(top_n=top_n, sort_failing_designs_last=True)

    for design in designs:
      self._experiment_design_evaluator.evaluate_design(
          design,
          n_aa_simulations=n_aa_simulations,
          n_ab_simulations=n_ab_simulations,
          with_progress_bar=with_progress_bar,
          overwrite_mode="extend",
          exp_start_date=self.exp_start_date,
      )

  def extend_design_by_id(
      self,
      design_id: str | list[str],
      n_aa_simulations: int | None = None,
      n_ab_simulations: int | None = None,
      with_progress_bar: bool = False,
  ) -> None:
    """Extends the design with the given ID with the given number of A/A and A/B simulations.

    Args:
      design_id: The ID of the design(s) to extend. Either a single design ID or
        a list of design IDs.
      n_aa_simulations: The number of extra A/A simulations to run. If None, the
        number of A/A simulations will be automatically determined to ensure
        statistically rigorous evaluations.
      n_ab_simulations: The number of extra A/B simulations to run. If None, the
        number of A/B simulations will be automatically determined to ensure
        statistically rigorous evaluations.
      with_progress_bar: Whether to show a progress bar while extending the
        design. Defaults to False. It's recommended to set this to True only if
        you are not also printing info logs to the console.
    """
    if isinstance(design_id, str):
      design_ids = [design_id]
    else:
      design_ids = design_id

    for design_id in design_ids:
      design = self.get_design_by_id(design_id)
      if design is None:
        logger.warning("Design with ID %s not found, skipping.", design_id)
        continue

      self._experiment_design_evaluator.evaluate_design(
          design,
          n_aa_simulations=n_aa_simulations,
          n_ab_simulations=n_ab_simulations,
          with_progress_bar=with_progress_bar,
          overwrite_mode="extend",
          exp_start_date=self.exp_start_date,
      )
