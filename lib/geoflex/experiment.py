"""The main experiment class for GeoFleX."""

from typing import Any
import geoflex.data
import geoflex.experiment_design
import pandas as pd

ExperimentDesign = geoflex.experiment_design.ExperimentDesign
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesignSpec = geoflex.experiment_design.ExperimentDesignSpec


class Experiment:
  """The main experiment class for GeoFleX."""

  def __init__(
      self,
      name: str,
      historical_data: GeoPerformanceDataset,
      design_spec: ExperimentDesignSpec,
  ):
    """Initializes the experiment.

    Args:
      name: The name of the experiment. This should be a short, descriptive but
        unique name for the experiment. It will be used when saving and loading
        from Google Drive, so it must be unique.
      historical_data: The historical data for the experiment.
      design_spec: The specification for the experiment.
    """
    self.name = name
    self.historical_data = historical_data
    self.design_spec = design_spec
    self.runtime_data = None
    self.experiment_start_date = None

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
