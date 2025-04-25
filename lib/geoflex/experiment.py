"""The main experiment class for GeoFleX."""

import functools
import logging
import warnings
from geoflex import utils
import geoflex.bootstrap
import geoflex.data
import geoflex.evaluation
import geoflex.experiment_design
import numpy as np
import optuna as op
import pandas as pd
import pydantic
from scipy import stats

ExperimentDesign = geoflex.experiment_design.ExperimentDesign
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesignSpec = geoflex.experiment_design.ExperimentDesignSpec
MultivariateTimeseriesBootstrap = (
    geoflex.bootstrap.MultivariateTimeseriesBootstrap
)
GeoAssignmentRepresentivenessScorer = (
    geoflex.evaluation.GeoAssignmentRepresentivenessScorer
)
EffectScope = geoflex.experiment_design.EffectScope
ParquetDataFrame = utils.ParquetDataFrame

logger = logging.getLogger(__name__)


def _format_experiment_budget(value, budget_type):
  """Formats the experiment budget into a single string for printing."""
  if budget_type == "percentage_change":
    return f"{value:.0%}"
  elif budget_type == "daily_budget":
    return f"${value} per day"
  elif budget_type == "total_budget":
    return f"${value} total"
  else:
    return f"Value = {value}, Type = {budget_type}"


def _format_geo_eligibility(control, treatment, exclude, **kwargs):
  """Creates separate columns for the geo eligibility for printing."""
  del kwargs

  out = {"geo_eligible_control": control, "geo_eligible_exclude": exclude}
  for i, treatment_cell in enumerate(treatment):
    out[f"geo_eligible_treatment_{i}"] = treatment_cell

  out_filtered = {k: v for k, v in out.items() if v}
  return pd.Series(out_filtered)


def _format_n_geos_per_group(control, treatment, exclude, **kwargs):
  """Creates three columns for the number of geos per group for printing."""
  del kwargs

  out = {"n_geos_control": len(control), "n_geos_exclude": len(exclude)}
  for i, treatment_cell in enumerate(treatment):
    out[f"n_geos_treatment_{i}"] = len(treatment_cell)

  return pd.Series(out)


class MaxTrialsCallback:
  """Used to stop an Optuna study after a certain number of trials.

  With the way we are using Optuna, some trials can be invalid and will return
  inf for the first value. We want to run Optuna for a certain number of
  valid trials, so we use this callback to stop the study after we reach that
  number.
  """

  def __init__(self, max_trials: int):
    """Initializes the callback.

    Args:
      max_trials: The maximum number of valid trials to run.
    """
    self.max_trials = max_trials

  def __call__(self, study: op.Study, trial: op.Trial) -> None:
    n_non_inf_trials = len([
        t
        for t in study.trials
        if t.values is not None and not np.isinf(t.values[0])
    ])

    if n_non_inf_trials >= self.max_trials:
      logger.info("Stopping study after %s trials.", n_non_inf_trials)
      study.stop()


class RawExperimentSimulationResults(pydantic.BaseModel):
  """The results of a simulation of an experiment design."""

  design: ExperimentDesign
  raw_eval_metrics: ParquetDataFrame
  representiveness_score: float
  primary_metric_standard_error: float

  model_config = pydantic.ConfigDict(
      extra="forbid",
      arbitrary_types_allowed=True,
  )

  @pydantic.model_validator(mode="after")
  def check_design_id_is_in_raw_eval_metrics(
      self,
  ) -> "RawExperimentSimulationResults":
    """Sets the design id in raw eval metrics if it is not already present."""
    if "design_id" not in self.raw_eval_metrics.columns:
      self.raw_eval_metrics["design_id"] = self.design.design_id
    return self


class Experiment(pydantic.BaseModel):
  """The main experiment class for GeoFleX.

  Attributes:
      name: The name of the experiment. This should be a short, descriptive but
        unique name for the experiment. It will be used when saving and loading
        from Google Drive, so it must be unique.
      historical_data: The historical data for the experiment.
      design_spec: The specification for the experiment.
      bootstrapper_seasons_per_block: The number of seasons per block for the
        bootstrapper.
      bootstrapper_log_transform: Whether to log transform the data for the
        bootstrapper. Only do this if all your metrics are non-negative.
      bootstrapper_seasonality: The seasonality for the bootstrapper. Defaults
        to 7 which assumes you have daily data with a weekly seasonality.
      runtime_data: The runtime data for the experiment. This will be used for
        the experiment analysis and will be added once the experiment is
        complete - it is not available before the experiment starts.
      experiment_start_date: The start date of the experiment. Defaults to None
        and can be set once the experiment has begun.
      experiment_simulation_results: The results of the experiment simulations.
        This will be populated as the simulations are run.
      validation_check_threhold: The threshold for failing a validation check.
        Defaults to 0.001, which is a 99.9% confidence level. Typically this
        does not need to be changed.
      selected_design_id: The ID of the selected design. This is the design that
        will be used for the experiment.
      study: The Optuna study for the experiment design exploration.
      pareto_front_design_ids: The design IDs that are on the Pareto front of
        the Optuna study. These are the designs that have the best combination
        of power and representativeness.
  """

  name: str
  historical_data: GeoPerformanceDataset
  design_spec: ExperimentDesignSpec
  bootstrapper_seasons_per_block: int = 2
  bootstrapper_log_transform: bool = True
  bootstrapper_seasonality: int = 7

  study: op.Study | None = None
  pareto_front_design_ids: list[str] = []
  experiment_simulation_results: dict[str, RawExperimentSimulationResults] = {}
  selected_design_id: str | None = None

  runtime_data: ParquetDataFrame | None = None
  experiment_start_date: str | None = None

  # 99.9% confidence level for failing a validation check
  validation_check_threhold: float = 0.001

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
    """Clears the designs."""
    logger.info("Clearing designs for experiment %s.", self.name)
    self.experiment_simulation_results = {}

  def record_design(
      self,
      design: ExperimentDesign,
      raw_eval_metrics: pd.DataFrame,
      primary_metric_standard_error: float,
      representiveness_score: float,
  ) -> None:
    """Records the design and its evaluation results.

    Args:
      design: The experiment design.
      raw_eval_metrics: The raw evaluation metrics for the design.
      primary_metric_standard_error: The standard error of the primary metric.
      representiveness_score: The representiveness score of the design.
    """
    self.experiment_simulation_results[design.design_id] = (
        RawExperimentSimulationResults(
            design=design,
            raw_eval_metrics=raw_eval_metrics,
            representiveness_score=representiveness_score,
            primary_metric_standard_error=primary_metric_standard_error,
        )
    )

  def get_experiment_design_results(
      self, design_id: str
  ) -> RawExperimentSimulationResults:
    """Returns the experiment design results for a given design_id."""
    return self.experiment_simulation_results[design_id]

  @property
  def all_raw_eval_metrics(self) -> pd.DataFrame:
    """Returns all raw evaluation metrics for all experiment designs."""
    raw_eval_metrics_list = [
        results.raw_eval_metrics
        for results in self.experiment_simulation_results.values()
    ]
    return pd.concat(raw_eval_metrics_list, ignore_index=True)

  @property
  def n_experiment_designs(self) -> int:
    """Returns the number of experiment designs."""
    return len(self.experiment_simulation_results)

  @functools.cached_property
  def bootstrapper(self) -> MultivariateTimeseriesBootstrap:
    """The bootstrapper for the experiment.

    This will return bootstrapped samples of the historical data, which can be
    used to estimate the standard error of the primary metric.
    """
    bootstrapper = MultivariateTimeseriesBootstrap(
        log_transform=self.bootstrapper_log_transform,
        seasonality=self.bootstrapper_seasonality,
        seasons_per_block=self.bootstrapper_seasons_per_block,
    )
    bootstrapper.fit(
        self.historical_data.pivoted_data,
        seasons_per_filt=self.bootstrapper_seasons_per_block,
    )
    return bootstrapper

  @functools.cached_property
  def representativeness_scorer(self) -> GeoAssignmentRepresentivenessScorer:
    """The representativeness scorer for the experiment.

    This evaluates how representative evaluate experiment designs if the
    effect scope is "all_geos".
    """
    return GeoAssignmentRepresentivenessScorer(
        historical_data=self.historical_data.parsed_data,
        geo_column_name=self.historical_data.geo_id_column,
        geos=self.historical_data.geos,
    )

  def suggest_experiment_design(self, trial: op.Trial) -> ExperimentDesign:
    """Suggests an experiment design for the given trial.

    It will suggest experiment designs based on the experiment design spec.

    Args:
      trial: The Optuna trial to use to suggest the experiment design.

    Returns:
      The suggested experiment design.
    """
    runtime_weeks = trial.suggest_categorical(
        "runtime_weeks",
        self.design_spec.runtime_weeks_candidates,
    )
    methodology_name = trial.suggest_categorical(
        "methodology", list(self.design_spec.eligible_methodologies)
    )
    geo_eligibility_id = trial.suggest_categorical(
        "geo_eligibility_id",
        list(range(len(self.design_spec.geo_eligibility_candidates))),
    )
    cell_volume_constraint_id = trial.suggest_categorical(
        "cell_volume_constraint_id",
        list(range(len(self.design_spec.cell_volume_constraint_candidates))),
    )
    random_seed = trial.suggest_categorical(
        "random_seed", self.design_spec.random_seeds
    )
    experiment_budget_id = trial.suggest_categorical(
        "experiment_budget_id",
        list(range(len(self.design_spec.experiment_budget_candidates))),
    )

    geo_eligibility = self.design_spec.geo_eligibility_candidates[
        geo_eligibility_id
    ]
    cell_volume_constraint = self.design_spec.cell_volume_constraint_candidates[
        cell_volume_constraint_id
    ]
    experiment_budget = self.design_spec.experiment_budget_candidates[
        experiment_budget_id
    ]

    methodology = geoflex.methodology.get_methodology(methodology_name)
    methodology_parameters = methodology.suggest_methodology_parameters(
        self.design_spec, trial
    )

    design = ExperimentDesign(
        experiment_type=self.design_spec.experiment_type,
        primary_metric=self.design_spec.primary_metric,
        experiment_budget=experiment_budget,
        secondary_metrics=self.design_spec.secondary_metrics,
        methodology=methodology_name,
        methodology_parameters=methodology_parameters,
        runtime_weeks=runtime_weeks,
        n_cells=self.design_spec.n_cells,
        alpha=self.design_spec.alpha,
        alternative_hypothesis=self.design_spec.alternative_hypothesis,
        geo_eligibility=geo_eligibility,
        cell_volume_constraint=cell_volume_constraint,
        random_seed=random_seed,
        effect_scope=self.design_spec.effect_scope,
    )

    return design

  def simulate_experiments(
      self, design: ExperimentDesign, simulations_per_trial: int
  ) -> pd.DataFrame | None:
    """Simulates experiments for the given design.

    Args:
      design: The experiment design to simulate.
      simulations_per_trial: The number of simulations to run.

    Returns:
      The results of the simulations, or None if the design is not eligible for
      the methodology.
    """
    methodology = geoflex.methodology.get_methodology(design.methodology)
    if not methodology.is_eligible_for_design(design):
      return None

    max_date = self.historical_data.parsed_data[
        self.historical_data.date_column
    ].max()
    exp_start_date = max_date - pd.Timedelta(
        weeks=max(self.design_spec.runtime_weeks_candidates)
    )

    # Cost-per-metric metrics must be inverted for power calculation
    # because we will run an A/A test, and the metric impact is 0, which
    # makes cost-per-metric metrics undefined. We will re-invert the results
    # at the end to get to the MDE for the original metric.
    if design.primary_metric.cost_per_metric:
      inverted_primary_metric = design.primary_metric.invert()
    else:
      inverted_primary_metric = design.primary_metric

    inverted_secondary_metrics = [
        metric.invert() if metric.cost_per_metric else metric
        for metric in design.secondary_metrics
    ]

    results_list = []
    random_seed = design.random_seed
    for sample in self.bootstrapper.sample_dataframes(simulations_per_trial):
      # Design the experiment with the pre-test data
      is_pretest = sample.index.values < exp_start_date
      pretest_dataset = GeoPerformanceDataset.from_pivoted_data(
          sample.loc[is_pretest],
          geo_id_column=self.historical_data.geo_id_column,
          date_column=self.historical_data.date_column,
      )

      # Different random seed for each sample
      random_seed += 1
      geo_assignment = methodology.assign_geos(
          design,
          pretest_dataset,
          np.random.default_rng(random_seed),
      )
      sample_design = design.model_copy(
          update={
              "geo_assignment": geo_assignment,
              "primary_metric": inverted_primary_metric,
              "secondary_metrics": inverted_secondary_metrics,
              "random_seed": random_seed,
          }
      )

      # Create the runtime dataset from the bootstrapped sample, simulate the \
      # experiment in it, and analyse it.
      runtime_dataset = GeoPerformanceDataset.from_pivoted_data(
          sample,
          geo_id_column=self.historical_data.geo_id_column,
          date_column=self.historical_data.date_column,
      ).simulate_experiment(
          experiment_start_date=exp_start_date,
          design=sample_design,
      )

      results_list.append(
          methodology.analyze_experiment(
              runtime_dataset,
              sample_design,
              exp_start_date.strftime("%Y-%m-%d"),
          )
      )

    results = pd.concat(results_list)
    return results

  def evaluate_single_simulation_results(self, data: pd.DataFrame) -> pd.Series:
    """Evaluates the results of a single metric simulation.

    Args:
      data: The results of a single simulation, as a DataFrame. Generated by the
        `simulate_experiments()` function.

    Returns:
      A Series with the evaluation results.
    """
    n_rows = len(data)
    has_relative = np.all(~data["point_estimate_relative"].isna())
    metric_name = data["metric"].values[0]

    avg_absolute_effect = data["point_estimate"].mean()
    unbiased_absolute_effect_pval = stats.ttest_1samp(
        data["point_estimate"], 0
    ).pvalue
    standard_error_absolute_effect = data["point_estimate"].std()
    covered_absolute_effect = (
        data["lower_bound"] * data["upper_bound"] < 0.0
    ).sum()
    coverage_absolute_effect = covered_absolute_effect / n_rows
    coverage_absolute_effect_pval = stats.binomtest(
        covered_absolute_effect,
        n=n_rows,
        p=1.0 - self.design_spec.alpha,
        alternative="less",
    ).pvalue

    absolute_effect_is_unbiased = (
        unbiased_absolute_effect_pval >= self.validation_check_threhold
    )
    absolute_effect_has_coverage = (
        coverage_absolute_effect_pval >= self.validation_check_threhold
    )

    all_checks_pass = (
        absolute_effect_is_unbiased and absolute_effect_has_coverage
    )
    failing_checks = []
    if not absolute_effect_is_unbiased:
      failing_checks.append(f"Biased absolute effect estimate ({metric_name})")
    if not absolute_effect_has_coverage:
      failing_checks.append(
          "Absolute effect confidence intervals don't meet target coverage"
          f" ({metric_name})"
      )

    if has_relative:
      relative_effects = data["point_estimate_relative"].astype(float)
      relative_lower_bounds = data["lower_bound_relative"].astype(float)
      relative_upper_bounds = data["upper_bound_relative"].astype(float)

      avg_relative_effect = relative_effects.mean()
      unbiased_relative_effect_pval = stats.ttest_1samp(
          relative_effects, 0
      ).pvalue
      standard_error_relative_effect = relative_effects.std()
      covered_relative_effect = (
          relative_lower_bounds * relative_upper_bounds < 0.0
      ).sum()
      coverage_relative_effect = covered_relative_effect / n_rows
      coverage_relative_effect_pval = stats.binomtest(
          covered_relative_effect,
          n=n_rows,
          p=1.0 - self.design_spec.alpha,
          alternative="less",
      ).pvalue

      relative_effect_is_unbiased = (
          unbiased_relative_effect_pval >= self.validation_check_threhold
      )
      relative_effect_has_coverage = (
          coverage_relative_effect_pval >= self.validation_check_threhold
      )

      all_checks_pass = (
          all_checks_pass
          and relative_effect_is_unbiased
          and relative_effect_has_coverage
      )

      if not relative_effect_is_unbiased:
        failing_checks.append(
            f"Biased relative effect estimate ({metric_name})"
        )
      if not relative_effect_has_coverage:
        failing_checks.append(
            "Relative effect confidence intervals don't meet target coverage"
            f" ({metric_name})"
        )

    else:
      avg_relative_effect = pd.NA
      standard_error_relative_effect = pd.NA
      coverage_relative_effect = pd.NA
      relative_effect_is_unbiased = pd.NA
      relative_effect_has_coverage = pd.NA

    return pd.Series(
        dict(
            avg_absolute_effect=avg_absolute_effect,
            standard_error_absolute_effect=standard_error_absolute_effect,
            coverage_absolute_effect=coverage_absolute_effect,
            absolute_effect_is_unbiased=absolute_effect_is_unbiased,
            absolute_effect_has_coverage=absolute_effect_has_coverage,
            avg_relative_effect=avg_relative_effect,
            standard_error_relative_effect=standard_error_relative_effect,
            coverage_relative_effect=coverage_relative_effect,
            relative_effect_is_unbiased=relative_effect_is_unbiased,
            relative_effect_has_coverage=relative_effect_has_coverage,
            all_checks_pass=all_checks_pass,
            failing_checks=failing_checks,
        )
    )

  def _design_evaluation_objective(
      self,
      trial: op.Trial,
      simulations_per_trial: int,
      ignore_designs_with_failing_checks: bool,
  ) -> tuple[float, float]:
    """Suggests a new experiment design, and evaluates it.

    This is the objective function for optuna. It will suggest a new experiment
    design based on the design spec, simulate the experiment, and evaluate the
    results. The evaluation will determine if the design is eligible for the
    methodology, and if the primary metric meets the validation checks.

    If the design is not eligible for the methodology, or if the primary metric
    does not meet the validation checks and ignore_designs_with_failing_checks,
    we will return inf for the standard error, and -1.0 for the
    representiveness score. This will ensure that this design is never selected.

    Args:
      trial: The Optuna trial to use to suggest the experiment design.
      simulations_per_trial: The number of simulations to run per trial. The
        siumulations are used to calculate the standard error of the effect
        size.
      ignore_designs_with_failing_checks: Whether to ignore designs with failing
        checks. If this is set to true, if the design fails the checks, we will
        return inf for the standard error, and -1.0 for the representiveness
        score. This will ensure that this design is never selected. If set to
        false, we will return the standard error and representiveness score,
        regardless of whether the checks pass.

    Returns:
      A tuple of the standard error of the primary metric, and the
      representiveness score.
    """
    # Suggest the experiment design based on the design spec.
    design = self.suggest_experiment_design(trial)
    trial.set_user_attr("design_id", design.design_id)

    # Simulate the experiment with the suggested design.
    results = self.simulate_experiments(design, simulations_per_trial)

    # If the design is not eligible for the methdology, the results will be
    # None. In this case we return inf for the standard error, and -1.0 for the
    # representiveness score. This will ensure that this design is never
    # selected.
    if results is None:
      logger.info(
          "Design %s (trial %s) is not eligible for methodology %s",
          design.design_id,
          trial.number,
          design.methodology,
      )
      return np.inf, -1.0

    # Assign geos for the experiment.
    design.geo_assignment = geoflex.methodology.get_methodology(
        design.methodology
    ).assign_geos(
        design, self.historical_data, np.random.default_rng(design.random_seed)
    )

    # If the effect scope is all geos, we need to ensure that the treatment
    # geos are representative of the entire population. Otherwise we do not.
    needs_representiveness = design.effect_scope == EffectScope.ALL_GEOS
    if needs_representiveness:
      assignment = design.geo_assignment.make_geo_assignment_array(
          self.representativeness_scorer.geos
      )
      representiveness_score = self.representativeness_scorer(
          assignment=assignment, with_pvalue=False
      )[0]
    else:
      representiveness_score = 0.0

    # We evaluate the primary metric for the objective for optuna. This will
    # ensure that we are finding the best possible design for the primary
    # metric.
    primary_metric_results = (
        results.loc[results["is_primary_metric"]]
        .groupby("cell")
        .apply(self.evaluate_single_simulation_results, include_groups=False)
    )

    # If the primary metric is a cost-per-metric metric, or a metric-per-cost
    # metric, we will use the standard error of the absolute effect. Otherwise,
    # we will use the standard error of the relative effect. These are the
    # easiest to interpret for the user.
    if (
        design.primary_metric.cost_per_metric
        or design.primary_metric.metric_per_cost
    ):
      primary_metric_standard_error = primary_metric_results[
          "standard_error_absolute_effect"
      ].max()
    else:
      primary_metric_standard_error = primary_metric_results[
          "standard_error_relative_effect"
      ].max()

    # Record the design results for later analysis.
    self.record_design(
        design=design,
        raw_eval_metrics=results,
        primary_metric_standard_error=primary_metric_standard_error,
        representiveness_score=representiveness_score,
    )

    # If the primary metric does not meet the validation checks, we return inf
    # for the standard error, and -1.0 for the representiveness score. This will
    # ensure that this design is never selected. The validation checks are
    # designed to ensure that the confidence intervals for the primary metric
    # are correctly calibrated and the effect size estimates are unbiased.
    primary_metric_all_checks_pass = primary_metric_results[
        "all_checks_pass"
    ].all()
    if not primary_metric_all_checks_pass:
      logger.info(
          "Design %s (trial %s) does not meet the validation checks for the"
          " primary metric, the following checks failed: %s",
          design.design_id,
          trial.number,
          primary_metric_results["failing_checks"].aggregate(
              lambda x: ", ".join(list(set(sum(x, []))))
          ),
      )
      if ignore_designs_with_failing_checks:
        return np.inf, -1.0

    # Finally we return the standard error of the primary metric, and the
    # representiveness score. These will be used by optuna to optimise the
    # objective. We will perform multi-objective optimisation to find the
    # smallest standard error and highest representiveness score.
    return primary_metric_standard_error, representiveness_score

  def _design_counting_evaluation_objective(
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
    design = self.suggest_experiment_design(trial)
    methodology = geoflex.methodology.get_methodology(design.methodology)
    if methodology.is_eligible_for_design(design):
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
    previous_verbosity = op.logging.get_verbosity()
    op.logging.set_verbosity(logging.CRITICAL)  # Disable logging.
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
        self._design_counting_evaluation_objective,
        n_trials=10_000,
        n_jobs=-1,
    )

    all_results = counting_study.trials_dataframe()

    counts = (
        all_results.groupby("user_attrs_methodology")["value"]
        .sum()
        .astype(int)
        .to_dict()
    )
    op.logging.set_verbosity(previous_verbosity)  # Restore logging.
    return counts

  def explore_experiment_designs(
      self,
      max_trials: int = 100,
      simulations_per_trial: int = 300,
      n_jobs: int = -1,
      seed: int = 0,
      ignore_designs_with_failing_checks: bool = True,
      warm_start: bool = True,
  ) -> None:
    """Explores how the different eligible experiment designs perform.

    Given your data and the experiment design spec, this function will explore
    how the different eligible experiment designs perform. The results can then
    be retrieved using the `get_top_designs()` function.

    Args:
      max_trials: The maximum number of trials to run. If there are more than
        max_trials eligible experiment designs, they will be randomly sampled.
      simulations_per_trial: The number of simulations to run for each eligible
        experiment design. This is used to estimate the performance of the
        experiment designs.
      n_jobs: The number of jobs to use for parallelisation. If set to -1, it
        will use all available CPU cores.
      seed: The random seed to use for sampling the experiment designs. This is
        used to ensure that the results are reproducible.
      ignore_designs_with_failing_checks: Whether to ignore designs with failing
        checks. If this is set to true, designs with failing checks will not be
        included in the results, and the exploration will look for max_trials of
        designs with passing checks. If set to false, the exploration will look
        for max_trials of designs, regardless of whether the checks pass.
      warm_start: Whether to warm start the exploration. If set to true, the
        exploration will warm start from the previous best designs. If set to
        false, the exploration will start from scratch.
    """

    objective = lambda trial: self._design_evaluation_objective(
        trial,
        simulations_per_trial=simulations_per_trial,
        ignore_designs_with_failing_checks=ignore_designs_with_failing_checks,
    )

    if warm_start and self.study is None and self.n_experiment_designs > 0:
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

    with warnings.catch_warnings():
      # Hide the experiment warning about the BruteForceSampler being
      # experimental.
      warnings.filterwarnings(
          "ignore", message="BruteForceSampler is experimental"
      )
      if self.study is None or not warm_start:
        logger.info("Creating new study.")
        self.study = op.create_study(
            sampler=op.samplers.BruteForceSampler(seed=seed),
            directions=[
                "minimize",
                "maximize",
            ],  # Minimise standard error, maximise representiveness
        )
      else:
        logger.info("Continuing existing study.")

    self.study.optimize(
        objective,
        n_trials=max_trials * 10,
        callbacks=[MaxTrialsCallback(max_trials)],
        n_jobs=n_jobs,
    )

    best_trial_numbers = [trial.number for trial in self.study.best_trials]
    self.pareto_front_design_ids = (
        self.study.trials_dataframe()
        .loc[best_trial_numbers, "user_attrs_design_id"]
        .values.tolist()
    )

  def _summarise_single_design_results(
      self,
      data: pd.DataFrame,
      use_relative_effects_where_possible: bool,
      target_power: float,
  ) -> pd.Series:
    """Summarises the results of a single design.

    This returns the MDEs for each metric, and the standard error of the
    primary metric. It also returns the failing checks as a list and whether
    all checks pass.

    Args:
      data: The data to summarise. This contains one row per metric for the
        design.
      use_relative_effects_where_possible: Whether to use the relative effects
        where possible. If the metric is a cost-per-metric or metric-per-cost
        metric, we will always use the absolute effect. Otherwise, if this is
        set to true we will use relative effects.
      target_power: The target power to use for the MDE calculations.

    Returns:
      A pandas series containing the MDEs for each metric, and the standard
      error of the primary metric, whether the validation checks pass and the
      list of failing checks.
    """
    output = {
        "failing_checks": [],
        "all_checks_pass": True,
    }

    for _, row in data.iterrows():
      metric_name = row["metric"]

      if use_relative_effects_where_possible and np.isfinite(
          row["relative_effect_standard_error"]
      ):
        standard_error = row["relative_effect_standard_error"]
        mde_name_prefix = "Relative "
      else:
        standard_error = row["absolute_effect_standard_error"]
        mde_name_prefix = ""

      if row["is_primary_metric"]:
        output["primary_metric_failing_checks"] = row["failing_checks"]
        output["primary_metric_all_checks_pass"] = row["all_checks_pass"]
        output["primary_metric_standard_error"] = standard_error
        primary_metric_mde_name = ", primary metric"
      else:
        primary_metric_mde_name = ""

      mde = geoflex.evaluation.calculate_minimum_detectable_effect_from_stats(
          standard_error=standard_error,
          alternative=self.design_spec.alternative_hypothesis,
          power=target_power,
          alpha=self.design_spec.alpha,
      )

      if "__INVERTED__" in metric_name:
        metric_name = metric_name.replace(" __INVERTED__", "")
        mde = 1.0 / mde

      output[
          f"{mde_name_prefix}MDE ({metric_name}{primary_metric_mde_name})"
      ] = mde
      output["failing_checks"] += row["failing_checks"]
      output["all_checks_pass"] &= row["all_checks_pass"]

    return pd.Series(output).sort_index()

  def _get_design_summaries(self) -> pd.DataFrame:
    """Returns summaries of the experiment designs as a dataframe."""
    out = (
        pd.DataFrame([
            design_results.design.model_dump()
            for design_results in self.experiment_simulation_results.values()
        ])
        .set_index("design_id")
        .drop(
            columns=[
                "experiment_type",
                "primary_metric",
                "secondary_metrics",
                "n_cells",
                "alpha",
                "alternative_hypothesis",
                "effect_scope",
            ]
        )
    )

    out["experiment_budget"] = out["experiment_budget"].apply(
        lambda x: _format_experiment_budget(**x)
    )

    out = (
        out["geo_eligibility"]
        .apply(lambda x: _format_geo_eligibility(**x))
        .join(out.drop(columns="geo_eligibility"))
    )
    out = (
        out["geo_assignment"]
        .apply(lambda x: _format_n_geos_per_group(**x))
        .join(out.drop(columns=["geo_assignment", "cell_volume_constraint"]))
    )
    return out

  def get_all_design_summaries(
      self,
      target_power: float = 0.8,
      target_primary_metric_mde: float | None = None,
      pareto_front_only: bool = False,
      include_design_parameters: bool = False,
      use_relative_effects_where_possible: bool = True,
  ) -> pd.DataFrame:
    """Returns all the experiment design summaries as a dataframe.

    If the experiment is a multi-cell experiment, then the performance metrics
    (the MDEs, standard errors, coverages and failing checks) are taken to be
    the worst case scenario for each cell. So the standard error is the maximum
    standard error across all cells, and the coverage is the minimum coverage
    across all cells, etc. This is to ensure that the best design selected is
    one that works across all cells in the design.

    If the effect scope is all geos, then the representiveness score is also
    calculated. A higher score is better, and a score close to 0
    is ideal. Worst case is a score of -1.

    Args:
      target_power: The target power to use for the MDE calculations. The MDE
        will be calculated for every metric specified in the design spec.
      target_primary_metric_mde: The target MDE for the primary metric. If this
        is set, then the power will be calculated for the primary metric at this
        MDE.
      pareto_front_only: Whether to only include the pareto front of the
        experiment designs. The pareto front is the set of designs that have the
        best performance on both representativeness and performance metrics.
      include_design_parameters: Whether to include the design parameters in the
        output. If not included just the design id and the results will be
        included.
      use_relative_effects_where_possible: Whether to use the relative effects
        where possible. For each metric, if the metric is a cost-per-metric or
        metric-per-cost metric, we will always use the absolute effect.
        Otherwise, if this is set to true we will use relative effects.

    Returns:
      A dataframe containing the experiment design summaries.
    """
    if target_primary_metric_mde is not None:
      error_message = (
          "Calculating the Power given an MDE is not yet implemented"
      )
      logger.error(error_message)
      raise NotImplementedError(error_message)
    raw_data = self.all_raw_eval_metrics

    if pareto_front_only:
      raw_data = raw_data.loc[
          raw_data["design_id"].isin(self.pareto_front_design_ids)
      ]

    # The design summary metrics are calculated by evaluating each simulation
    # result for each metric, and then aggregating the results across the cells
    # taking the maximum standard error and minimum coverage (i.e. the worst
    # case scenario for each cell).
    design_summary_metrics = (
        raw_data.groupby(["design_id", "metric", "is_primary_metric", "cell"])[[
            "metric",
            "point_estimate",
            "point_estimate_relative",
            "lower_bound",
            "upper_bound",
            "lower_bound_relative",
            "upper_bound_relative",
        ]]
        .apply(self.evaluate_single_simulation_results)
        .reset_index()
        .groupby(["design_id", "is_primary_metric", "metric"])
        .agg(
            unbiased_absolute_effect=("absolute_effect_is_unbiased", "all"),
            unbiased_relative_effect=("relative_effect_is_unbiased", "all"),
            coverage_absolute_effect=("absolute_effect_has_coverage", "min"),
            coverage_relative_effect=("relative_effect_has_coverage", "min"),
            all_checks_pass=("all_checks_pass", "all"),
            absolute_effect_standard_error=(
                "standard_error_absolute_effect",
                "max",
            ),
            relative_effect_standard_error=(
                "standard_error_relative_effect",
                "max",
            ),
            failing_checks=("failing_checks", lambda x: list(set(sum(x, [])))),
        )
        .reset_index()
        .groupby("design_id")
        .apply(
            self._summarise_single_design_results,
            use_relative_effects_where_possible=use_relative_effects_where_possible,
            target_power=target_power,
            include_groups=False,
        )
    )

    if include_design_parameters:
      design_summary_metrics = self._get_design_summaries().join(
          design_summary_metrics,
          how="inner",
      )

    if self.design_spec.effect_scope == EffectScope.ALL_GEOS:
      design_summary_metrics["treatment_groups_representiveness_score"] = (
          pd.Series({
              design_id: values.representiveness_score
              for design_id, values in self.experiment_simulation_results.items()
          })
      )

    return design_summary_metrics.sort_values("primary_metric_standard_error")

  def select_design(self, design_id: str) -> None:
    """Sets the selected design, based on the design_id."""
    self.selected_design_id = design_id

  def get_selected_design(self) -> ExperimentDesign:
    """Returns the selected design.

    Raises:
      ValueError: If no design has been selected, or if the selected design
      does not exist.
    """
    if self.selected_design_id is None:
      error_message = "No design has been selected."
      logger.error(error_message)
      raise ValueError(error_message)
    if self.selected_design_id not in self.experiment_simulation_results:
      error_message = f"Design {self.selected_design_id} does not exist."
      logger.error(error_message)
      raise ValueError(error_message)
    return self.experiment_simulation_results[self.selected_design_id].design

  def save_to_file(self, file_path: str) -> None:
    """Saves the experiment to a file."""
    raise NotImplementedError()

  @classmethod
  def load_from_file(cls, file_path: str) -> "Experiment":
    """Loads the experiment from a file."""
    raise NotImplementedError()

  def save_to_google_drive(self) -> str:
    """Saves the experiment to a Google Drive folder.

    If this experiment has not previously been saved to Google Drive, it will
    be saved to a new folder. Otherwise, it will be saved to the existing folder
    that it was previously saved to.

    If this experiment has not been saved to Google Drive before, but a folder
    with the same name already exists, it will be saved to a new folder with the
    name "{name} (1)".

    Returns:
      The URL of the Google Drive folder that the experiment was saved to.
    """
    raise NotImplementedError()

  @classmethod
  def load_from_google_drive(cls, drive_folder_url: str) -> "Experiment":
    """Loads the experiment from a Google Drive folder.

    Args:
      drive_folder_url: The URL of the Google Drive folder to load the
        experiment from.
    """
    raise NotImplementedError()

  def set_runtime_data(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_start_date: str,
  ) -> None:
    """Sets the runtime data for the experiment.

    Args:
      runtime_data: The runtime data for the experiment. Must include the
        treatment and control geos, the required pre-experiment period, and
        either partial or full runtime period.
      experiment_start_date: The start date of the experiment. This is the date
        that the treatment geos were first exposed to the treatment.

    Raises:
      ValueError: If the runtime data does not include the treatment and control
      geos, if the runtime data does not include the required pre-experiment
      period, or if the experiment start date is before the final date in the
      historical data.
    """
    raise NotImplementedError()

  def can_run_full_analysis(self) -> bool:
    """Returns whether it is possible to fully analyse this experiment.

    It is only possible to fully analyse an experiment if the runtime data
    includes the full runtime period.
    """
    raise NotImplementedError()

  def run_partial_analysis(self) -> None:
    """Runs a partial analysis on the experiment.

    This includes simple timeseries plots of the response metric for the
    treatment and control geos. It is used to monitor ongoing experiments, and
    will not return the final results of the experiment.
    """
    raise NotImplementedError()

  def run_full_analysis(self) -> None:
    """Runs a full analysis on the experiment.

    This will return the final results of the experiment, including the effect
    sizes, p-values and confidence intervals or all key metrics.
    """
    raise NotImplementedError()
