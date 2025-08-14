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

"""GeoFleX: A flexible and unified geo-experiment solution."""

import logging
from geoflex import data
from geoflex import evaluation
from geoflex import experiment_design
from geoflex import exploration_spec
from geoflex import explore
from geoflex import methodology
from geoflex import metrics
from geoflex import visualization
import optuna as op

ExperimentDesignExplorationSpec = (
    exploration_spec.ExperimentDesignExplorationSpec
)
GeoPerformanceDataset = data.GeoPerformanceDataset
GeoEligibility = experiment_design.GeoEligibility
GeoAssignment = experiment_design.GeoAssignment
ExperimentDesign = experiment_design.ExperimentDesign
ExperimentDesignEvaluator = evaluation.ExperimentDesignEvaluator
ExperimentDesignEvaluationResults = evaluation.ExperimentDesignEvaluationResults
ExperimentDesignExplorer = explore.ExperimentDesignExplorer
ExperimentBudget = experiment_design.ExperimentBudget
ExperimentBudgetType = experiment_design.ExperimentBudgetType
CellVolumeConstraint = experiment_design.CellVolumeConstraint
CellVolumeConstraintType = experiment_design.CellVolumeConstraintType
EffectScope = experiment_design.EffectScope
Methodology = methodology.Methodology
register_methodology = methodology.register_methodology
list_methodologies = methodology.list_methodologies
assign_geos = methodology.assign_geos
analyze_experiment = methodology.analyze_experiment
design_is_eligible_for_data = methodology.design_is_eligible_for_data
compare_designs = experiment_design.compare_designs
display_analysis_results = visualization.display_analysis_results

__version__ = "0.3.0"

logger = logging.getLogger(__name__)

op.logging.disable_default_handler()
op.logging.set_verbosity(logging.WARNING)  # Default to warning log level.


def start_logging_to_stdout(level: int = logging.INFO) -> None:
  """Starts logging to stdout for geoflex.

  Args:
    level: The logging level to set.
  """
  logging.basicConfig(
      level=level,
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      force=True,
      handlers=[logging.StreamHandler()],
  )
  op.logging.enable_default_handler()
  op.logging.set_verbosity(level)

  logger.info("Logging enabled, GeoFleX version: %s", __version__)


def stop_logging_to_stdout() -> None:
  """Stops logging to stdout for geoflex."""
  logger.info("Stopping logging to stdout.")

  logging.basicConfig(force=True, handlers=[])
  op.logging.disable_default_handler()
