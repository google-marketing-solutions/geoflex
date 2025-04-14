"""The main experiment class for GeoFleX."""

import functools
from typing import Any
import geoflex.bootstrap
import geoflex.data
import geoflex.evaluation
import geoflex.experiment_design
import numpy as np
import optuna as op
import pandas as pd

ExperimentDesign = geoflex.experiment_design.ExperimentDesign
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesignSpec = geoflex.experiment_design.ExperimentDesignSpec
MultivariateTimeseriesBootstrap = (
    geoflex.bootstrap.MultivariateTimeseriesBootstrap
)
GeoAssignmentRepresentivenessScorer = (
    geoflex.evaluation.GeoAssignmentRepresentivenessScorer
)


class Experiment:
  """The main experiment class for GeoFleX."""

  def __init__(
      self,
      name: str,
      historical_data: GeoPerformanceDataset,
      design_spec: ExperimentDesignSpec,
      bootstrapper_seasons_per_block: int = 2,
      bootstrapper_log_transform: bool = True,
      bootstrapper_seasonality: int = 7,
  ):
    """Initializes the experiment.

    Args:
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
    """
    self.name = name
    self.historical_data = historical_data
    self.design_spec = design_spec
    self.runtime_data = None
    self.experiment_start_date = None

    self.bootstrapper_seasons_per_block = bootstrapper_seasons_per_block
    self.bootstrapper_log_transform = bootstrapper_log_transform
    self.bootstrapper_seasonality = bootstrapper_seasonality

    self._drive_folder_url = None  # Will be set when saving to Google Drive.
    self.clear_designs()

  def clear_designs(self) -> None:
    """Clears the designs."""
    self._eligible_experiment_design_results = {}
    self._selected_design_id = None

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
    self._eligible_experiment_design_results[design.design_id] = {
        "design": design,
        "raw_eval_metrics": raw_eval_metrics,
        "representiveness_score": representiveness_score,
        "primary_metric_standard_error": primary_metric_standard_error,
    }
    self._eligible_experiment_design_results[design.design_id][
        "raw_eval_metrics"
    ]["design_id"] = design.design_id

  def get_experiment_design_results(self, design_id: str) -> dict[str, Any]:
    """Returns the experiment design results for a given design_id."""
    return self._eligible_experiment_design_results[design_id]

  @property
  def all_raw_eval_metrics(self) -> pd.DataFrame:
    """Returns all raw evaluation metrics for all experiment designs."""
    raw_eval_metrics_list = [
        results["raw_eval_metrics"]
        for results in self._eligible_experiment_design_results.values()
    ]
    return pd.concat(raw_eval_metrics_list, ignore_index=True)

  @property
  def n_experiment_designs(self) -> int:
    """Returns the number of experiment designs."""
    return len(self._eligible_experiment_design_results)

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
    runtime_weeks = trial.suggest_int(
        "runtime_weeks",
        self.design_spec.min_runtime_weeks,
        self.design_spec.max_runtime_weeks,
    )
    methodology_name = trial.suggest_categorical(
        "methodology", list(self.design_spec.eligible_methodologies)
    )
    geo_eligibility_id = trial.suggest_categorical(
        "geo_eligibility_id",
        list(range(len(self.design_spec.geo_eligibility_candidates))),
    )
    n_geos_per_group_id = trial.suggest_categorical(
        "n_geos_per_group_id",
        list(range(len(self.design_spec.n_geos_per_group_candidates))),
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
    n_geos_per_group = self.design_spec.n_geos_per_group_candidates[
        n_geos_per_group_id
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
        n_geos_per_group=n_geos_per_group,
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
        weeks=self.design_spec.max_runtime_weeks
    )

    design.geo_assignment = methodology.assign_geos(
        design, self.historical_data, np.random.default_rng(design.random_seed)
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
      pretest_dataset = geoflex.GeoPerformanceDataset.from_pivoted_data(
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
      runtime_dataset = geoflex.GeoPerformanceDataset.from_pivoted_data(
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

  def explore_experiment_designs(self, max_trials: int = 100) -> None:
    """Explores how the different eligible experiment designs perform.

    Given your data and the experiment design spec, this function will explore
    how the different eligible experiment designs perform. The results can then
    be retrieved using the `get_top_designs()` function.

    Args:
      max_trials: The maximum number of trials to run. If there are more than
        max_trials eligible experiment designs, they will be randomly sampled.
    """

    raise NotImplementedError()

  def get_top_designs(self) -> list[ExperimentDesign]:
    """Returns the top experiment designs."""
    raise NotImplementedError()

  def select_design(self, design_id: str) -> None:
    """Sets the selected design, based on the design_id."""
    raise NotImplementedError()

  def get_selected_design(self) -> ExperimentDesign:
    """Returns the selected design.

    Raises:
      ValueError: If no design has been selected, or if the selected design
      does not exist.
    """
    if self._selected_design_id is None:
      raise ValueError("No design has been selected.")
    if self._selected_design_id not in self._eligible_experiment_design_results:
      raise ValueError(f"Design {self._selected_design_id} does not exist.")
    return self._eligible_experiment_design_results[self._selected_design_id][
        "design"
    ]

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
