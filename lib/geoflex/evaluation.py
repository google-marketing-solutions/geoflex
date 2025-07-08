"""Methods to evaluate the quality of a geo experiments."""

import functools
import logging
from typing import Any
import geoflex.bootstrap
import geoflex.data
import geoflex.experiment_design
import geoflex.methodology
import geoflex.metrics
import geoflex.utils
import numpy as np
import pandas as pd
import pydantic
from scipy import stats
import tqdm.auto as tqdm


logger = logging.getLogger(__name__)

ExperimentDesign = geoflex.experiment_design.ExperimentDesign
ParquetDataFrame = geoflex.utils.ParquetDataFrame
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
MultivariateTimeseriesBootstrap = (
    geoflex.bootstrap.MultivariateTimeseriesBootstrap
)
Metric = geoflex.metrics.Metric
ExperimentDesignEvaluationResults = (
    geoflex.experiment_design.ExperimentDesignEvaluationResults
)
SingleEvaluationResult = geoflex.experiment_design.SingleEvaluationResult
EffectScope = geoflex.experiment_design.EffectScope
GeoAssignment = geoflex.experiment_design.GeoAssignment
CellVolumeConstraint = geoflex.experiment_design.CellVolumeConstraint
CellVolumeConstraintType = geoflex.experiment_design.CellVolumeConstraintType


class GeoAssignmentRepresentativenessScorer:
  """Scores the representativeness of a geo assignment.

  Ideally, each of the treatment groups in a geo assignment should be
  representative ofthe entire population. This scorer calculated the average
  Pearson correlation between all pairs of geos in the population. Then for a
  given assignment, it calculates the average similarity between all of the
  geos and the closest treatment geo.

  A good assignment will have a high score, close to 1.0. A low score indicates
  that the treatment groups are not representative of the entire population.

  Example usage:
  ```
    scorer = GeoAssignmentRepresentativenessScorer(
        historical_data=historical_data.pivoted_data,
        geos=geos,
    )
    assignment = np.array([0,0,1,1])
    score = scorer(assignment)
  ```
  """

  def __init__(
      self,
      historical_data: pd.DataFrame,
      geos: list[str],
  ):
    """Initializes the RepresentativenessScorer.

    Args:
      historical_data: The data to score, should come from the pivoted data in
        the GeoPerformanceDataset. It will have a column for each geo and
        metric.
      geos: The geos to be included in the assignment. If None, all geos in the
        historical data will be included. The order is important, the order of
        the assignment must match the order of the geos.
    """
    self.historical_data = historical_data
    self.geos = geos

  @functools.cached_property
  def similarity_matrix(self) -> np.array:
    """The similarity matrix for the geos.

    This is calculated as the average Pearson correlation between all pairs of
    geos. A score of 1 indicates that the geos are identical, a score of -1
    indicates they are perfectly opposite.

    Returns:
      The similarity matrix for the geos. Rows / columns are ordered by the
      order of self.geos.

    Raises:
      RuntimeError: If the similarity matrix contains non-finite values. This
        is unexpected.
    """

    # Initialize an empty NumPy array to store the similarity scores
    n_geos = len(self.geos)
    similarity_matrix = np.zeros((n_geos, n_geos))

    corr_matrix = self.historical_data.swaplevel(axis=1).corr()

    for i, geo1 in enumerate(self.geos):
      for j, geo2 in enumerate(self.geos[i:], start=i):

        if i == j:
          # If they are thr same geo, then they are identical, skip the
          # computation
          similarity_matrix[i, j] = 1.0
          continue

        avg_similarity = np.diag(corr_matrix.loc[geo1, geo2].values).mean()
        similarity_matrix[i, j] = avg_similarity
        similarity_matrix[j, i] = avg_similarity

    if not np.all(np.isfinite(similarity_matrix)):
      error_message = (
          "Similarity matrix contains non-finite values, this is unexpected."
      )
      logger.error(error_message)
      raise RuntimeError(error_message)

    return similarity_matrix

  def __call__(self, assignment: np.ndarray) -> float:
    """Calculates the representativeness score for the given assignment.

    This is calculated as the average similarity between all of the geos and the
    closest treatment geo. A score close to 1 indicates a representative
    assignment, while a score close to 0 or negative indicates a
    non-representative assignment.

    Args:
      assignment: The geo assignment to score. This is an array of 0s and 1s,
        where 0 indicates a control geo and 1 indicates a treatment geo.

    Returns:
      The representativeness score for the given assignment.
    """
    n_samples = self.similarity_matrix.shape[0]
    if len(assignment) != n_samples:
      error_message = (
          "Assignment has length {len(assignment)}, but the similarity matrix"
          " has length {n_samples}."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    if not np.all((assignment == 1) | (assignment == 0)):
      error_message = (
          "Assignment contains values other than 0 or 1, this is unexpected."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    is_treatment = assignment == 1
    if not np.any(is_treatment):
      # If there are no treatment geos, then the assignment is not
      # representative
      return -1.0

    # Find the maximum value for each geo to the best treatment geo, and take
    # the average for the score.
    return self.similarity_matrix[is_treatment, :].max(axis=0).mean()


def calculate_minimum_detectable_effect_from_stats(
    standard_error: float,
    alternative: str = "two-sided",
    power: float = 0.8,
    alpha: float = 0.05,
) -> float:
  """Calculates the minimum detectable effect from the standard error.

  The minimum detectable effect is the smallest true effect size that would
  return a statistically significant result, at the specified alpha level
  and with the specified alternative hypothesis, [power]% of the time.

  The returned minimum detectable effect is always positive, even if the
  alternative hypothesis is "less".

  Args:
    standard_error: The standard error of the test statistic.
    alternative: The alternative hypothesis being tested, one of ['two-sided',
      'greater', 'less'].
    power: The desired statistical power, as a fraction. Defaults to 0.8, which
      would mean a power of 80%.
    alpha: The alpha level of the test, defaults to 0.05.

  Returns:
    The minimum detectable absolute effect size.

  Raises:
     ValueError: If the alternative is not one of ['two-sided', 'greater',
      'less'].
  """
  if alternative == "two-sided":
    z_alpha = stats.norm.ppf(q=1.0 - alpha / 2.0)
  elif alternative in ["greater", "less"]:
    z_alpha = stats.norm.ppf(q=1.0 - alpha)
  else:
    error_message = (
        "Alternative must be one of ['two-sided', 'greater', 'less']"
    )
    logger.error(error_message)
    raise ValueError(error_message)

  z_power = stats.norm.ppf(power)

  return standard_error * (z_alpha + z_power)


class RawExperimentSimulationResults(pydantic.BaseModel):
  """The results of a simulation of an experiment design.

  Attributes:
    design: The experiment design that was simulated.
    simulation_results: The simulation results for the experiment design.
    representativeness_scores: The representativeness scores for the experiment
      design, of each treatment cell.
    design_is_valid: Whether the design is valid or not. The design is not valid
      if the methodology is not eligible for the design.
    warnings: A list of warnings that were generated during the simulation.
  """

  design: ExperimentDesign
  simulation_results: ParquetDataFrame
  representativeness_scores: list[float] | None
  design_is_valid: bool
  sufficient_simulations: bool
  warnings: list[str] = []

  model_config = pydantic.ConfigDict(
      extra="forbid",
      arbitrary_types_allowed=True,
  )

  @pydantic.model_validator(mode="after")
  def set_design_id_in_simulation_results(
      self,
  ) -> "RawExperimentSimulationResults":
    """Sets the design id in simulation results."""
    if not self.design_is_valid:
      # Simulation results are empty if the design is not valid.
      return self

    self.simulation_results["design_id"] = self.design.design_id
    return self

  @pydantic.model_validator(mode="after")
  def check_results_for_all_cells_exist(
      self,
  ) -> "RawExperimentSimulationResults":
    """Checks that the results for all cells exist."""
    if not self.design_is_valid:
      # Simulation results are empty if the design is not valid.
      return self

    n_treatment_cells = self.design.n_cells - 1
    if n_treatment_cells != self.simulation_results["cell"].nunique():
      error_message = (
          f"The simulation results for design {self.design.design_id} do not"
          f" contain results for all cells. Expected {self.design.n_cells},"
          f" got {len(self.simulation_results.index)}."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    if n_treatment_cells != len(self.representativeness_scores):
      error_message = (
          f"The simulation results for design {self.design.design_id} do not"
          " contain representativeness scores for all cells. Expected"
          f" {self.design.n_cells}, got {len(self.representativeness_scores)}."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    return self


def validate_cell_volume_constraint_is_respected(
    geo_assignment: GeoAssignment,
    cell_volume_constraint: CellVolumeConstraint,
    historical_data: GeoPerformanceDataset,
) -> tuple[CellVolumeConstraint | None, bool, str]:
  """Validates that the cell volume constraint is respected.

  Args:
    geo_assignment: The geo assignment to validate.
    cell_volume_constraint: The cell volume constraint to validate.
    historical_data: The historical data to use for the validation.

  Returns:
    A list of the cell volumes where the first element is the control cell
    volume, and the rest are the treatment cell volumes, a boolean indicating
    whether the constraint is respected, and an error message if the constraint
    is not respected. If there are no constraints, the cell volumes are None.
  """
  if all(value is None for value in cell_volume_constraint.values):
    # If there are no constraints, it always passes.
    actual_cell_volumes = None
    is_valid = True
    error_message = ""
    return actual_cell_volumes, is_valid, error_message

  if (
      cell_volume_constraint.constraint_type
      == CellVolumeConstraintType.MAX_GEOS
  ):
    # Cell volume is the number of geos in each cell.
    actual_cell_volumes = [len(geo_assignment.control)]
    for treatment_group in geo_assignment.treatment:
      actual_cell_volumes.append(len(treatment_group))

  elif cell_volume_constraint.constraint_type == (
      CellVolumeConstraintType.MAX_PERCENTAGE_OF_METRIC
  ):
    # Cell volume is the percentage of the metric in each cell.
    data = historical_data.parsed_data.copy()
    data["geo_assignment"] = geo_assignment.make_geo_assignment_array(
        data[historical_data.geo_id_column].values
    )
    cell_volumes = data.groupby("geo_assignment")[
        cell_volume_constraint.metric_column
    ].sum()
    cell_volumes /= cell_volumes.sum()

    actual_cell_volumes = [
        float(cell_volumes.loc[i])
        for i in range(len(geo_assignment.treatment) + 1)
    ]
  else:
    raise ValueError(
        "Unsupported cell volume constraint type:"
        f" {cell_volume_constraint.constraint_type}"
    )

  error_message = ""
  is_valid = True
  actual_cell_volumes = CellVolumeConstraint(
      constraint_type=cell_volume_constraint.constraint_type,
      metric_column=cell_volume_constraint.metric_column,
      values=actual_cell_volumes,
  )

  for actual_cell_volume, constraint_value in zip(
      actual_cell_volumes.values, cell_volume_constraint.values
  ):
    if constraint_value is None:
      continue

    if actual_cell_volume > constraint_value:
      is_valid = False
      error_message = (
          "Cell volume constraint is not respected. Target ="
          f" {cell_volume_constraint}, Actual = {actual_cell_volumes}."
      )
      break

  return actual_cell_volumes, is_valid, error_message


def estimate_validation_check_minimum_sample_size(
    p_null: float,
    p_delta: float,
    alpha: float,
    power: float = 0.8,
) -> int:
  """Estimates the minimum sample size for the validation check.

  The validation checks are proportion checks, performed with a binomial test.
  The minimum sample size is estimated using the normal approximation of the
  binomial distribution.

  The validation check is a one-sided test, and the direction is inferred based
  on the sign of the delta.

  Args:
    p_null: The proportion for the null hypothesis.
    p_delta: The change in proportion for the alternative hypothesis.
    alpha: The alpha level of the test.
    power: The power level of the test.

  Returns:
    The minimum sample size.
  """
  p_alt = np.clip(p_null + p_delta, a_min=0.0, a_max=1.0)

  z_alpha = stats.norm.ppf(1 - alpha)
  z_beta = stats.norm.ppf(power)

  numerator = (
      z_alpha * np.sqrt(p_null * (1 - p_null))
      + z_beta * np.sqrt(p_alt * (1 - p_alt))
  ) ** 2
  denominator = (p_alt - p_null) ** 2
  n_analytical = numerator / denominator

  # The sample size must be an integer, so we round up.
  return int(np.ceil(n_analytical))


class ExperimentDesignEvaluator(pydantic.BaseModel):
  """The class for evaluating experiment designs.

  Attributes:
    historical_data: The historical data to use for the experiment design.
    simulations_per_trial: The number of simulations to run per trial. If None,
      the number of simulations will be automatically selected to ensure that
      there are enough simulations to meet the minimum sample size required for
      the validation checks.
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
    representativeness_scorer_metrics: The column names of the metrics to be
      used for the representativeness scorer. If None, all metrics in the
      historical data will be used. The representativeness is calculated by
      looking at the Pearson correlation across these metrics between the
      control and treatment groups.
    validation_check_threhold: The threhold to use for the validation check.
      Defaults to 0.001, which is a 99.9% confidence level. Typically this does
      not need to be changed.
    raw_simulation_results: The raw simulation results for the experiment
      design. This is a dictionary of design_id to
      RawExperimentSimulationResults. Every time a new design is evaluated, the
      results will be added to this dictionary.
    experiment_design_evaluation_results: The experiment design evaluation
      results for the experiment design. This is a dictionary of design_id to
      ExperimentDesignEvaluationResults. Every time a new design is evaluated,
      the results will be added to this dictionary.
  """

  historical_data: GeoPerformanceDataset
  simulations_per_trial: int | None = None

  bootstrapper_seasons_per_block: int = 4
  bootstrapper_model_type: str = "multiplicative"
  bootstrapper_seasonality: int = 7
  bootstrapper_sampling_type: str = "permutation"
  bootstrapper_stl_params: dict[str, Any] | None = None

  representativeness_scorer_metrics: list[str] | None = None

  validation_check_threhold: float = 0.001

  raw_simulation_results: dict[str, RawExperimentSimulationResults] = {}
  experiment_design_evaluation_results: dict[
      str, ExperimentDesignEvaluationResults
  ] = {}

  @functools.cached_property
  def bootstrapper(self) -> MultivariateTimeseriesBootstrap:
    """The bootstrapper for the experiment.

    This will return bootstrapped samples of the historical data, which can be
    used to estimate the standard error of the primary metric.
    """
    bootstrapper = MultivariateTimeseriesBootstrap(
        model_type=self.bootstrapper_model_type,
        seasonality=self.bootstrapper_seasonality,
        seasons_per_block=self.bootstrapper_seasons_per_block,
        sampling_type=self.bootstrapper_sampling_type,
        stl_params=self.bootstrapper_stl_params,
    )
    bootstrapper.fit(
        self.historical_data.pivoted_data,
    )
    return bootstrapper

  @functools.cached_property
  def representativeness_scorer(self) -> GeoAssignmentRepresentativenessScorer:
    """The representativeness scorer for the experiment.

    This evaluates how representative evaluate experiment designs if the
    effect scope is "all_geos".
    """
    data = self.historical_data.pivoted_data
    if self.representativeness_scorer_metrics is not None:
      data = data[self.representativeness_scorer_metrics]

    return GeoAssignmentRepresentativenessScorer(
        historical_data=data,
        geos=self.historical_data.geos,
    )

  def get_aa_coverage_minimum_sample_size(self, alpha: float) -> int:
    """The minimum sample size for the A/A coverage check.

    Args:
      alpha: The alpha level of the test.

    Returns:
      The minimum sample size required to test the coverage of the A/A test,
      given the alpha level.
    """
    return estimate_validation_check_minimum_sample_size(
        p_null=1.0 - alpha,
        p_delta=-0.1,
        alpha=self.validation_check_threhold,
        power=0.8,
    )

  def _set_exp_start_date(
      self,
      design: ExperimentDesign,
      exp_start_date: pd.Timestamp | None = None,
  ) -> pd.Timestamp:
    """Sets the experiment start date for the given design."""
    max_date = self.historical_data.parsed_data[
        self.historical_data.date_column
    ].max()
    min_date = self.historical_data.parsed_data[
        self.historical_data.date_column
    ].min()

    latest_exp_start_date = max_date - pd.Timedelta(weeks=design.runtime_weeks)
    earliest_exp_start_date = min_date + pd.Timedelta(
        weeks=design.runtime_weeks
    )

    days_of_data = (max_date - min_date).days + 1
    weeks_of_data = days_of_data / 7

    if latest_exp_start_date < earliest_exp_start_date:
      error_message = (
          f"The runtime weeks {design.runtime_weeks} is too long for the"
          " available historical data. We require at least double the runtime"
          " weeks of historical data to run the evaluation, but there are only"
          f" {days_of_data} days ({weeks_of_data:.2f} weeks) of historical"
          " data."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    if exp_start_date is None:
      exp_start_date = latest_exp_start_date

    if exp_start_date > latest_exp_start_date:
      error_message = (
          f"The experiment start date {exp_start_date:%Y-%m-%d} for the"
          " simulation is after the latest feasible date in the historical"
          f" data {latest_exp_start_date:%Y-%m-%d}. This can happen if the"
          " runtime weeks are too long for the available historical data."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    if exp_start_date < earliest_exp_start_date:
      error_message = (
          f"The experiment start date {exp_start_date:%Y-%m-%d} for the"
          " simulation is before the earliest feasible date in the historical"
          f" data {earliest_exp_start_date:%Y-%m-%d}. This can happen if the"
          " runtime weeks are too long for the available historical data."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    return exp_start_date

  def _invert_cost_per_metric_metrics(
      self, design: ExperimentDesign
  ) -> ExperimentDesign:
    """Inverts cost-per-metric metrics for power calculation.

    Cost-per-metric metrics must be inverted for power calculation
    because we will run an A/A test, and the metric impact is 0, which
    makes cost-per-metric metrics undefined. We will re-invert the results
    at the end to get to the MDE for the original metric.

    All other metrics are left unchanged.

    Args:
      design: The experiment design to evaluate.

    Returns:
      A new experiment design with the inverted cost-per-metric metrics.
    """
    if design.primary_metric.cost_per_metric:
      inverted_primary_metric = design.primary_metric.invert()
    else:
      inverted_primary_metric = design.primary_metric

    inverted_secondary_metrics = [
        metric.invert() if metric.cost_per_metric else metric
        for metric in design.secondary_metrics
    ]

    return design.model_copy(
        update={
            "primary_metric": inverted_primary_metric,
            "secondary_metrics": inverted_secondary_metrics,
        },
        deep=True,
    )

  def _assign_geos_with_pretest_sample_data(
      self,
      sample: pd.DataFrame,
      design: ExperimentDesign,
      exp_start_date: pd.Timestamp,
  ) -> None:
    """Assigns geos based on the methodology and the pretest sample data."""
    is_pretest = sample.index.values < exp_start_date
    pretest_dataset = GeoPerformanceDataset.from_pivoted_data(
        sample.loc[is_pretest],
        geo_id_column=self.historical_data.geo_id_column,
        date_column=self.historical_data.date_column,
    )

    geoflex.methodology.assign_geos(design, pretest_dataset)

  def _analyze_simulated_experiment(
      self,
      sample: pd.DataFrame,
      design: ExperimentDesign,
      exp_start_date: pd.Timestamp,
  ):
    """Simulates an experiment and analyzes it."""
    runtime_dataset = GeoPerformanceDataset.from_pivoted_data(
        sample,
        geo_id_column=self.historical_data.geo_id_column,
        date_column=self.historical_data.date_column,
    ).simulate_experiment(
        experiment_start_date=exp_start_date,
        design=design,
    )

    return geoflex.methodology.analyze_experiment(
        design,
        runtime_dataset,
        exp_start_date.strftime("%Y-%m-%d"),
    )

  def _evaluate_representativeness_if_all_geos_scope(
      self, design: ExperimentDesign
  ) -> list[float]:
    """Evaluates representativeness if the effect scope is all geos."""
    needs_representativeness = design.effect_scope == EffectScope.ALL_GEOS
    if not needs_representativeness:
      return [1.0] * (design.n_cells - 1)

    assignment = design.geo_assignment.make_geo_assignment_array(
        self.representativeness_scorer.geos
    )
    representativeness_scores = []
    for cell_number in range(1, design.n_cells):
      cell_assignment = (assignment == cell_number).astype(int)
      representativeness_score = self.representativeness_scorer(
          assignment=cell_assignment
      )
      representativeness_scores.append(representativeness_score)

    return representativeness_scores

  def simulate_experiment_results(
      self,
      design: ExperimentDesign,
      n_simulations: int | None = None,
      exp_start_date: pd.Timestamp | None = None,
      with_progress_bar: bool = False,
  ) -> RawExperimentSimulationResults | None:
    """Simulates experiments for the given design.

    This creates n_simulations bootstrap samples of the historical data,
    and then for each sample it assigns the geos, simulates an experiment,
    analyzed the experiment, and records the results.

    If a metric is a cost-per-metric metric, then the results are inverted
    because we will run an A/A test, and the metric impact is 0, which
    makes cost-per-metric metrics undefined. We will re-invert the results
    at the end to get to the MDE for the original metric.

    If the design does not have a geo assignment, then it will create one.

    Args:
      design: The experiment design to evaluate.
      n_simulations: The number of simulations to run. If None, the number of
        simulations will be automatically selected to ensure that there are
        enough simulations to meet the minimum sample size required for the
        validation checks.
      exp_start_date: The start date to use to simulate the experiment. If None,
        the start date will be calculated from the maximum date in the
        historical data and the runtime weeks in the experiment design.
      with_progress_bar: Whether to show a progress bar. Defaults to False. It's
        recommended to set this to True only if you are not also printing info
        logs to the console.

    Returns:
      The results of the simulations, or None if the design is not eligible for
      the methodology.

    Raises:
      ValueError: If the experiment start date is after the latest feasible date
      in the historical data. The latest feasible date is calculated as the
      maximum date in the historical data minus the runtime weeks in the
      experiment design.
    """
    warnings = []

    min_simulations = self.get_aa_coverage_minimum_sample_size(
        alpha=design.alpha
    )
    if n_simulations is None:
      logger.debug(
          "Number of simulations set automatically to %s.", min_simulations
      )
      n_simulations = min_simulations
    else:
      logger.debug("Number of simulations set by user to %s.", n_simulations)

    sufficient_simulations = n_simulations >= min_simulations
    if not sufficient_simulations:
      warnings.append(
          f"The number of simulations ({n_simulations}) is less than the"
          " minimum required for conclusive validation checks"
          f" ({min_simulations})."
      )

    exp_start_date = self._set_exp_start_date(design, exp_start_date)

    is_pretest = (
        self.historical_data.parsed_data[self.historical_data.date_column]
        < exp_start_date
    )
    simulated_pretest_data = GeoPerformanceDataset(
        data=self.historical_data.parsed_data.loc[is_pretest],
        geo_id_column=self.historical_data.geo_id_column,
        date_column=self.historical_data.date_column,
    )

    if not geoflex.methodology.design_is_eligible_for_data(
        design, simulated_pretest_data
    ):
      results = RawExperimentSimulationResults(
          design=design,
          simulation_results=pd.DataFrame(),
          representativeness_scores=None,
          design_is_valid=False,
          warnings=warnings,
          sufficient_simulations=sufficient_simulations,
      )
      self.raw_simulation_results[design.design_id] = results
      return results

    # Assign geos if they are not already assigned.
    if design.geo_assignment is None:
      geoflex.methodology.assign_geos(design, self.historical_data)

    representativeness_scores = (
        self._evaluate_representativeness_if_all_geos_scope(design)
    )

    design_with_inverted_metrics = self._invert_cost_per_metric_metrics(design)

    results_list = []
    random_seed = design.random_seed

    iterator = self.bootstrapper.sample_dataframes(n_simulations)
    if with_progress_bar:
      iterator = tqdm.tqdm(
          iterator, total=n_simulations, desc=f"Evaluating {design.design_id}"
      )

    for sample in iterator:

      if geoflex.methodology.is_pseudo_experiment(design):
        # For pseudo experiments, we don't want to re-assign the geos for every
        # sample, so we will use the original geo assignment. This is because
        # the geo assignment for a pseudo experiment is typically not random,
        # or at least the validity of the experiment does not depend on the
        # randomisation of the geo assignment. Since the geo assignment can be
        # slow, we will skip reassigning the geos for each sample.
        sample_design = design_with_inverted_metrics.model_copy()
      else:
        # If it's not a pseudo experiment, we will reassign the geos for each
        # sample. This is because for a non-pseudo experiment the randomisation
        # of the geos is a key aspect of ensuring unbiased results.
        random_seed = np.random.default_rng(random_seed).integers(0, 2**31 - 1)
        sample_design = design_with_inverted_metrics.make_variation(
            random_seed=random_seed,
            geo_assignment=None,
        )
        self._assign_geos_with_pretest_sample_data(
            sample,
            sample_design,
            exp_start_date,
        )

      if sample_design.geo_assignment is None:
        logger.warning(
            "The original data was eligible for the design, but one of the"
            " bootstrap samples is not, because the geo assignment failed."
            " This might be an error with the bootstrap sampler or the"
            " methodology.",
        )
        results = RawExperimentSimulationResults(
            design=design,
            simulation_results=pd.DataFrame(),
            representativeness_scores=None,
            design_is_valid=False,
            warnings=warnings,
            sufficient_simulations=sufficient_simulations,
        )
        self.raw_simulation_results[design.design_id] = results
        return results

      sample_results = self._analyze_simulated_experiment(
          sample, sample_design, exp_start_date
      )
      if sample_results is None:
        logger.warning(
            "The original data was eligible for the design, but one of the"
            " bootstrap samples is not, because the analysis failed. This"
            " might be an error with the bootstrap sampler or the methodology."
        )
        results = RawExperimentSimulationResults(
            design=design,
            simulation_results=pd.DataFrame(),
            representativeness_scores=None,
            design_is_valid=False,
            warnings=warnings,
            sufficient_simulations=sufficient_simulations,
        )
        self.raw_simulation_results[design.design_id] = results
        return results

      results_list.append(sample_results)

    results = RawExperimentSimulationResults(
        design=design,
        simulation_results=pd.concat(results_list),
        representativeness_scores=representativeness_scores,
        design_is_valid=True,
        warnings=warnings,
        sufficient_simulations=sufficient_simulations
    )
    self.raw_simulation_results[design.design_id] = results
    return results

  def _estimate_standard_errors(
      self, data: pd.DataFrame
  ) -> tuple[float, float | None]:
    """Estimates the standard errors."""
    absolute_effect_standard_error = data["point_estimate"].std()

    has_relative = np.all(~data["point_estimate_relative"].isna())
    if has_relative:
      relative_effect_standard_error = data["point_estimate_relative"].std()
    else:
      relative_effect_standard_error = None

    return absolute_effect_standard_error, relative_effect_standard_error

  def _check_effect_estimates_are_unbiased(
      self, data: pd.DataFrame
  ) -> tuple[bool, bool | None]:
    """Checks if the absolute effect is unbiased."""
    unbiased_absolute_effect_pval = stats.ttest_1samp(
        data["point_estimate"], 0
    ).pvalue
    absolute_effect_is_unbiased = (
        unbiased_absolute_effect_pval >= self.validation_check_threhold
    )

    has_relative = np.all(~data["point_estimate_relative"].isna())
    if has_relative:
      unbiased_relative_effect_pval = stats.ttest_1samp(
          data["point_estimate_relative"].astype(float), 0
      ).pvalue
      relative_effect_is_unbiased = (
          unbiased_relative_effect_pval >= self.validation_check_threhold
      )
    else:
      relative_effect_is_unbiased = None

    return absolute_effect_is_unbiased, relative_effect_is_unbiased

  def _estimate_coverage(
      self, data: pd.DataFrame, alpha: float
  ) -> tuple[float, bool, float | None, bool | None]:
    """Estimates the coverage and whether it meets the target coverage."""
    n_rows = len(data)

    # Get coverage
    covered_absolute_effect = (
        data["lower_bound"] * data["upper_bound"] < 0.0
    ).sum()
    coverage_absolute_effect = covered_absolute_effect / n_rows

    # Check coverage meets target
    coverage_absolute_effect_pval = stats.binomtest(
        covered_absolute_effect,
        n=n_rows,
        p=1.0 - alpha,
        alternative="less",
    ).pvalue
    absolute_effect_has_coverage = (
        coverage_absolute_effect_pval >= self.validation_check_threhold
    )

    # Now check relative effect coverage if it exists
    has_relative = np.all(~data["point_estimate_relative"].isna())
    if has_relative:
      covered_relative_effect = (
          data["lower_bound_relative"] * data["upper_bound_relative"] < 0.0
      ).sum()
      coverage_relative_effect = covered_relative_effect / n_rows
      coverage_relative_effect_pval = stats.binomtest(
          covered_relative_effect,
          n=n_rows,
          p=1.0 - alpha,
          alternative="less",
      ).pvalue
      relative_effect_has_coverage = (
          coverage_relative_effect_pval >= self.validation_check_threhold
      )
    else:
      coverage_relative_effect = None
      relative_effect_has_coverage = None

    return (
        coverage_absolute_effect,
        absolute_effect_has_coverage,
        coverage_relative_effect,
        relative_effect_has_coverage,
    )

  def _summarise_checks(
      self,
      metric_name: str,
      absolute_effect_is_unbiased: bool,
      absolute_effect_has_coverage: bool,
      relative_effect_is_unbiased: bool | None,
      relative_effect_has_coverage: bool | None,
  ) -> tuple[bool, list[str]]:
    """Summarises the checks."""
    metric_name_uninverted = metric_name.replace("__INVERTED__", "")
    all_checks_pass = True
    failing_checks = []
    if not absolute_effect_is_unbiased:
      all_checks_pass = False
      failing_checks.append(
          f"Biased absolute effect estimate ({metric_name_uninverted})"
      )
    if not absolute_effect_has_coverage:
      all_checks_pass = False
      failing_checks.append(
          "Absolute effect confidence intervals don't meet target coverage"
          f" ({metric_name_uninverted})"
      )

    if (
        relative_effect_is_unbiased is not None
        and not relative_effect_is_unbiased
    ):
      all_checks_pass = False
      failing_checks.append(
          f"Biased relative effect estimate ({metric_name_uninverted})"
      )
    if (
        relative_effect_has_coverage is not None
        and not relative_effect_has_coverage
    ):
      all_checks_pass = False
      failing_checks.append(
          "Relative effect confidence intervals don't meet target"
          f" coverage ({metric_name_uninverted})"
      )

    return all_checks_pass, failing_checks

  def _evaluate_raw_simulation_results(
      self,
      raw_simulation_results: RawExperimentSimulationResults,
  ) -> ExperimentDesignEvaluationResults:
    """Evaluates the results of a single metric simulation.

    Args:
      raw_simulation_results: The results of a single simulation.

    Returns:
      A Series with the evaluation results.
    """
    design = raw_simulation_results.design

    other_errors = []
    if design.geo_assignment is not None:
      # Validate the cell volume constraint is respected if the assignment
      # has been made.
      (
          actual_cell_volumes,
          is_valid_cell_volume_constraint,
          cell_volume_constraint_error,
      ) = geoflex.evaluation.validate_cell_volume_constraint_is_respected(
          geo_assignment=design.geo_assignment,
          cell_volume_constraint=design.cell_volume_constraint,
          historical_data=self.historical_data,
      )
      other_errors.append(cell_volume_constraint_error)
    else:
      actual_cell_volumes = None
      is_valid_cell_volume_constraint = True

    is_valid_design = (
        is_valid_cell_volume_constraint
        and raw_simulation_results.design_is_valid
    )

    if not raw_simulation_results.design_is_valid:
      # If the design is not eligible for the methodology, then the results
      # are None, and we don't want to evaluate it. We just return the design
      # unchanged.
      results = ExperimentDesignEvaluationResults(
          primary_metric_name=design.primary_metric.name,
          all_metric_results_per_cell=None,
          alpha=design.alpha,
          alternative_hypothesis=design.alternative_hypothesis,
          representativeness_scores_per_cell=None,
          actual_cell_volumes=actual_cell_volumes,
          other_errors=other_errors,
          is_valid_design=False,
          warnings=raw_simulation_results.warnings,
          sufficient_simulations=raw_simulation_results.sufficient_simulations,
      )
      self.experiment_design_evaluation_results[design.design_id] = results
      return results

    simulation_results = raw_simulation_results.simulation_results
    all_metric_results_per_cell = {}
    for metric_name, metric_data in simulation_results.groupby("metric"):
      all_metric_results_per_cell[metric_name] = []
      for _, data in metric_data.sort_values("cell").groupby("cell"):
        standard_error_absolute_effect, standard_error_relative_effect = (
            self._estimate_standard_errors(data)
        )

        absolute_effect_is_unbiased, relative_effect_is_unbiased = (
            self._check_effect_estimates_are_unbiased(data)
        )

        (
            coverage_absolute_effect,
            absolute_effect_has_coverage,
            coverage_relative_effect,
            relative_effect_has_coverage,
        ) = self._estimate_coverage(data, design.alpha)

        all_checks_pass, failing_checks = self._summarise_checks(
            metric_name,
            absolute_effect_is_unbiased,
            absolute_effect_has_coverage,
            relative_effect_is_unbiased,
            relative_effect_has_coverage,
        )

        all_metric_results_per_cell[metric_name].append(
            SingleEvaluationResult(
                standard_error_absolute_effect=standard_error_absolute_effect,
                standard_error_relative_effect=standard_error_relative_effect,
                coverage_absolute_effect=coverage_absolute_effect,
                coverage_relative_effect=coverage_relative_effect,
                all_checks_pass=all_checks_pass,
                failing_checks=failing_checks,
            )
        )

    results = ExperimentDesignEvaluationResults(
        primary_metric_name=design.primary_metric.name,
        all_metric_results_per_cell=all_metric_results_per_cell,
        alpha=design.alpha,
        alternative_hypothesis=design.alternative_hypothesis,
        representativeness_scores_per_cell=raw_simulation_results.representativeness_scores,
        actual_cell_volumes=actual_cell_volumes,
        other_errors=other_errors,
        is_valid_design=is_valid_design,
        warnings=raw_simulation_results.warnings,
        sufficient_simulations=raw_simulation_results.sufficient_simulations,
    )
    self.experiment_design_evaluation_results[design.design_id] = results
    return results

  def evaluate_design(
      self,
      design: ExperimentDesign,
      exp_start_date: pd.Timestamp | None = None,
      add_to_design: bool = True,
      force_evaluation: bool = False,
      with_progress_bar: bool = False,
  ) -> ExperimentDesignEvaluationResults | None:
    """Evaluates the design.

    This will perform the following steps:
    1. Create n_simulations bootstrap samples of the historical data.
    2. For each sample, assign geos, simulate an experiment, and analyze it.
    3. Record the results of each simulation (bootstrap sample).
    4. Evaluate the results of the simulations to estimate the standard errors
       and check the coverage.

    It will record the raw simulation results in the `raw_simulation_results`
    dictionary attribute of this evaluator object, if needed for debugging.

    Args:
      design: The experiment design to evaluate.
      exp_start_date: The start date to use to simulate the experiment. If None,
        the start date will be calculated from the maximum date in the
        historical data and the runtime weeks in the experiment design.
      add_to_design: Whether to add the evaluation results to the experiment
        design.
      force_evaluation: Whether to force evaluation, even if the design already
        has evaluation results. If true, and add_to_design is true, the
        evaluation results will be overwritten.
      with_progress_bar: Whether to show a progress bar while evaluating the
        design. Defaults to False. It's recommended to set this to True only if
        you are not also printing info logs to the console.

    Returns:
      The experiment design evaluation results.
    """
    if design.evaluation_results is not None:
      if force_evaluation and add_to_design:
        logger.warning(
            "Forcing evaluation of design %s, even though it already has"
            " evaluation results. The existing results will be overwritten.",
            design.design_id,
        )
      elif force_evaluation:
        logger.warning(
            "Forcing evaluation of design %s, even though it already has"
            " evaluation results. The new results will not be added to the "
            "design - the existing results will remain.",
            design.design_id,
        )
        design.evaluation_results = None
      else:
        logger.info(
            "Design %s already has evaluation results, skipping evaluation and"
            " returning the existing results.",
            design.design_id,
        )
        return design.evaluation_results

    logger.info("Evaluating design %s.", design.design_id)

    raw_simulation_results = self.simulate_experiment_results(
        design,
        self.simulations_per_trial,
        exp_start_date,
        with_progress_bar=with_progress_bar,
    )
    evaluation_results = self._evaluate_raw_simulation_results(
        raw_simulation_results
    )

    if add_to_design:
      design.evaluation_results = evaluation_results

    logger.info("Evaluation completed for design %s.", design.design_id)
    return evaluation_results
