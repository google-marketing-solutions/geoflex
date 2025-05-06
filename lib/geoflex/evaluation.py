"""Methods to evaluate the quality of a geo experiments."""

import functools
import logging
import dcor
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
from sklearn.metrics import silhouette_score

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


class GeoAssignmentRepresentivenessScorer:
  """Scores the representativeness of a geo assignment.

  Ideally, each of the treatment groups in a geo assignment should be
  representative ofthe entire population. This scorer uses the silhouette score,
  in combination with distance correlation, to measure the representativeness of
  the treatment groups.

  1. The distance correlation is used to measure the "distance" between two
    geos. It is defined as `1 - the distance correlation coefficient`. This
    gives a distance matrix for all the geos. The distance correlation
    coefficient is a multivariate correlation which evaluates the similarity
    based on all numerical columns in the data. For more information see:
    https://arxiv.org/pdf/0803.4101
  2. The silhouette score is used to measure the "quality" of a geo assignment.
    It is defined as the maximum silhouette score for the treatment groups in
    the assignment. For more information see:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html
  3. The representiveness score is then calculated as -1 * the silhouette score.
    This is because a lower silhouette score indicates a better assignment.
  4. If applied, a permutation test is used to calculate a p-value to indicate
    the likelihood of the assignment being as representative as a random
    assignment.

  A good assignment will have a low silhouette score, ideally lower than 0. A
  high score indicates that the treatment groups are separate clusters and are
  therefore not representative of the entire population.

  Example usage:
  ```
    scorer = GeoAssignmentRepresentivenessScorer(
        historical_data=historical_data,
        geo_column_name="geo_id",
        geos=geos,
        silhouette_sample_size=100,
    )
    assignment = np.array([-1, 0, 0, 1, 1, 2, 2])
    # Fast calculation without a p-value.
    score, _ = scorer(assignment)
    # Slow calculation with a p-value.
    score, pvalue = scorer(assignment, with_pvalue=True)
  ```
  """

  def __init__(
      self,
      historical_data: pd.DataFrame,
      geo_column_name: str,
      geos: list[str],
      silhouette_sample_size: int | None = None,
  ):
    """Initializes the RepresentivenessScorer.

    Args:
      historical_data: The data to score. Must contain the geo column and the
        performance metrics. All numeric columns will be included in the
        representiveness scoring.
      geo_column_name: The column name of the geo column.
      geos: The geos to be included in the assignment. If None, all geos in the
        historical data will be included. The order is important, the order of
        the assignment must match the order of the geos.
      silhouette_sample_size: The sample size to use for the silhouette score.
        If None, the full data is used.
    """
    self.historical_data = historical_data
    self.geo_column_name = geo_column_name
    self.silhouette_sample_size = silhouette_sample_size
    self.geos = geos

  @functools.cached_property
  def distance_matrix(self) -> np.array:
    """The distance matrix for the geos.

    This is calculated as 1 - the distance correlation coefficient. For more
    information see: https://arxiv.org/pdf/0803.4101
    """
    distance_matrix = np.zeros((len(self.geos), len(self.geos)))
    for i, geo_i in enumerate(self.geos):
      for j, geo_j in enumerate(self.geos):
        if i < j:
          x = (
              self.historical_data.loc[
                  self.historical_data[self.geo_column_name] == geo_i
              ]
              .select_dtypes("number")
              .values
          )
          y = (
              self.historical_data.loc[
                  self.historical_data[self.geo_column_name] == geo_j
              ]
              .select_dtypes("number")
              .values
          )
          x_normed = (x - x.mean(axis=0)) / x.std(axis=0)
          y_normed = (y - y.mean(axis=0)) / y.std(axis=0)
          corr = dcor.distance_correlation(x_normed, y_normed)
          distance_matrix[i, j] = 1.0 - corr
          distance_matrix[j, i] = 1.0 - corr
    return distance_matrix

  def _permutation_samples(
      self, assignment: np.ndarray, n_samples: int
  ) -> np.ndarray:
    """Returns the permutation samples of the score."""
    assignment = assignment.copy()
    scores = np.empty(n_samples)
    for i in range(n_samples):
      np.random.shuffle(assignment)
      scores[i] = self._score(assignment)
    return scores

  def _score(self, assignment: np.ndarray) -> float:
    """Calculates the representativeness score for the given assignment.

    This is -1 * the maximum silhouette score for the treatment groups in the
    assignment.

    Args:
      assignment: The geo assignment to score. This is assumed to be a numpy
        array of integers, where -1 indicates an excluded geo, 0 indicates a
        control geo and 1+ indicates a treatment geo (there can be multiple
        treatment groups in a multi-cell test).

    Returns:
      The representativeness score for the given assignment.
    """
    max_assignment = np.max(assignment)
    scores = []
    for i in range(1, max_assignment + 1):
      treatment_vs_rest_assignment = assignment == i
      scores.append(
          silhouette_score(
              self.distance_matrix,
              treatment_vs_rest_assignment,
              metric="precomputed",
              sample_size=self.silhouette_sample_size,
          )
      )
    return -1 * np.max(scores)

  def __call__(
      self,
      assignment: np.ndarray,
      with_pvalue: bool = False,
      n_permutation_samples: int = 200,
  ) -> tuple[float, float | None]:
    """Scores the representativeness of the given geo assignment.

    The ideal score is close to 0.0, which indicates that the treatment groups
    are representative of the entire population. A negative score indicates that
    the treatment groups are separate clusters and are therefore not
    representative of the entire population. A highly positive score is usually
    just due to random chance, but can occur if you don't have many geos.

    Args:
      assignment: The geo assignment to score. This is assumed to be a numpy
        array of integers, where -1 indicates an excluded geo, 0 indicates a
        control geo and 1+ indicates a treatment geo (there can be multiple
        treatment groups in a multi-cell test).
      with_pvalue: If True, a p-value will be calculated using permutation
        testing. Note: this will make the scoring much slower.
      n_permutation_samples: The number of permutation samples to use for the
        permutation test.

    Returns:
      The score of the geo assignment and the p-value if with_pvalue is True,
      otherwise None.
    """
    score = self._score(assignment)
    if with_pvalue:
      permutation_scores = self._permutation_samples(
          assignment, n_permutation_samples
      )
      pvalue = np.mean(score > permutation_scores)
    else:
      pvalue = None
    return score, pvalue


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
    representiveness_scores: The representativeness scores for the experiment
      design, of each treatment cell.
    design_is_valid: Whether the design is valid or not. The design is not valid
      if the methodology is not eligible for the design.
  """

  design: ExperimentDesign
  simulation_results: ParquetDataFrame
  representiveness_scores: list[float] | None
  design_is_valid: bool

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

    if n_treatment_cells != len(self.representiveness_scores):
      error_message = (
          f"The simulation results for design {self.design.design_id} do not"
          " contain representativeness scores for all cells. Expected"
          f" {self.design.n_cells}, got {len(self.representiveness_scores)}."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    return self


class ExperimentDesignEvaluator(pydantic.BaseModel):
  """The class for evaluating experiment designs.

  Attributes:
    historical_data: The historical data to use for the experiment design.
    simulations_per_trial: The number of simulations to run per trial.
    bootstrapper_seasons_per_block: The number of seasons per block to use for
      the bootstrapper.
    bootstrapper_log_transform: Whether to log transform the data for the
      bootstrapper.
    bootstrapper_seasonality: The seasonality to use for the bootstrapper.
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
  simulations_per_trial: int = 100

  bootstrapper_seasons_per_block: int = 2
  bootstrapper_log_transform: bool = True
  bootstrapper_seasonality: int = 7

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

  def _set_exp_start_date(
      self,
      design: ExperimentDesign,
      exp_start_date: pd.Timestamp | None = None,
  ) -> pd.Timestamp:
    """Sets the experiment start date for the given design."""
    max_date = self.historical_data.parsed_data[
        self.historical_data.date_column
    ].max()
    latest_exp_start_date = max_date - pd.Timedelta(weeks=design.runtime_weeks)
    if exp_start_date is None:
      exp_start_date = latest_exp_start_date
    elif exp_start_date > latest_exp_start_date:
      error_message = (
          f"The experiment start date {exp_start_date:%Y-%m-%d} for the"
          " simulation is after the latest feasible date in the historical"
          f" data {latest_exp_start_date:%Y-%m-%d}. This can happen if the"
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

  def _evaluate_representiveness_if_all_geos_scope(
      self, design: ExperimentDesign
  ) -> list[float]:
    """Evaluates representativeness if the effect scope is all geos."""
    needs_representiveness = design.effect_scope == EffectScope.ALL_GEOS
    if not needs_representiveness:
      return [0.0] * (design.n_cells - 1)

    assignment = design.geo_assignment.make_geo_assignment_array(
        self.representativeness_scorer.geos
    )
    representiveness_scores = []
    for cell_number in range(1, design.n_cells):
      cell_assignment = (assignment == cell_number).astype(int)
      representiveness_score = self.representativeness_scorer(
          assignment=cell_assignment, with_pvalue=False
      )[0]
      representiveness_scores.append(representiveness_score)

    return representiveness_scores

  def simulate_experiment_results(
      self,
      design: ExperimentDesign,
      n_simulations: int,
      exp_start_date: pd.Timestamp | None = None,
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
      n_simulations: The number of simulations to run.
      exp_start_date: The start date to use to simulate the experiment. If None,
        the start date will be calculated from the maximum date in the
        historical data and the runtime weeks in the experiment design.

    Returns:
      The results of the simulations, or None if the design is not eligible for
      the methodology.

    Raises:
      ValueError: If the experiment start date is after the latest feasible date
      in the historical data. The latest feasible date is calculated as the
      maximum date in the historical data minus the runtime weeks in the
      experiment design.
    """
    if not geoflex.methodology.design_is_valid(design):
      results = RawExperimentSimulationResults(
          design=design,
          simulation_results=pd.DataFrame(),
          representiveness_scores=None,
          design_is_valid=False,
      )
      self.raw_simulation_results[design.design_id] = results
      return results

    if design.geo_assignment is None:
      geoflex.methodology.assign_geos(design, self.historical_data)

    representiveness_scores = self._evaluate_representiveness_if_all_geos_scope(
        design
    )

    exp_start_date = self._set_exp_start_date(design, exp_start_date)
    design_with_inverted_metrics = self._invert_cost_per_metric_metrics(design)

    results_list = []
    random_seed = design.random_seed
    for sample in self.bootstrapper.sample_dataframes(n_simulations):
      # New random seed for each sample
      random_seed += 1
      sample_design = design_with_inverted_metrics.model_copy(
          update={"random_seed": random_seed}, deep=True
      )

      self._assign_geos_with_pretest_sample_data(
          sample,
          sample_design,
          exp_start_date,
      )

      sample_results = self._analyze_simulated_experiment(
          sample, sample_design, exp_start_date
      )

      results_list.append(sample_results)

    results = RawExperimentSimulationResults(
        design=design,
        simulation_results=pd.concat(results_list),
        representiveness_scores=representiveness_scores,
        design_is_valid=True,
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

    if not raw_simulation_results.design_is_valid:
      # If the design is not eligible for the methodology, then the results
      # are None, and we don't want to evaluate it. We just return the design
      # unchanged.
      results = ExperimentDesignEvaluationResults(
          primary_metric_name=design.primary_metric.name,
          all_metric_results_per_cell=None,
          alpha=design.alpha,
          alternative_hypothesis=design.alternative_hypothesis,
          representiveness_scores_per_cell=None,
          is_valid_design=False,
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
        representiveness_scores_per_cell=raw_simulation_results.representiveness_scores,
        is_valid_design=True,
    )
    self.experiment_design_evaluation_results[design.design_id] = results
    return results

  def evaluate_design(
      self,
      design: ExperimentDesign,
      exp_start_date: pd.Timestamp | None = None,
      add_to_design: bool = True,
      force_evaluation: bool = False,
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
        design, self.simulations_per_trial, exp_start_date
    )
    evaluation_results = self._evaluate_raw_simulation_results(
        raw_simulation_results
    )

    if add_to_design:
      design.evaluation_results = evaluation_results

    logger.info("Evaluation completed for design %s.", design.design_id)
    return evaluation_results
