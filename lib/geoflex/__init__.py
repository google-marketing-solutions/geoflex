"""GeoFleX: A flexible and unified geo-experiment solution."""

from geoflex import data
from geoflex import evaluation
from geoflex import experiment
from geoflex import experiment_design
from geoflex import methodology
from geoflex import metrics

ExperimentDesignSpec = experiment_design.ExperimentDesignSpec
Methodology = methodology.Methodology
ExperimentType = experiment_design.ExperimentType
GeoPerformanceDataset = data.GeoPerformanceDataset
Experiment = experiment.Experiment
ExperimentDesign = experiment_design.ExperimentDesign
ExperimentBudget = experiment_design.ExperimentBudget
ExperimentBudgetType = experiment_design.ExperimentBudgetType
EffectScope = experiment_design.EffectScope
register_methodology = methodology.register_methodology
list_methodologies = methodology.list_methodologies

__version__ = "0.0.1"
