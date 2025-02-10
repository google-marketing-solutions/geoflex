"""GeoFleX: A flexible and unified geo-experiment solution."""

from geoflex import data
from geoflex import experiment
from geoflex import experiment_design
from geoflex import methodology

ExperimentDesignConstraints = experiment_design.ExperimentDesignConstraints
Methodology = methodology.Methodology
ExperimentType = experiment_design.ExperimentType
GeoPerformanceDataset = data.GeoPerformanceDataset
Experiment = experiment.Experiment
ExperimentDesign = experiment_design.ExperimentDesign

__version__ = "0.0.1"
