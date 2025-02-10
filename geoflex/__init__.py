"""GeoFleX: A flexible and unified geo-experiment solution."""

import geoflex.constraints
import geoflex.data
import geoflex.experiment

ExperimentDesignConstraints = geoflex.constraints.ExperimentDesignConstraints
Methodology = geoflex.constraints.Methodology
ExperimentType = geoflex.constraints.ExperimentType
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
Experiment = geoflex.experiment.Experiment
ExperimentDesign = geoflex.experiment.ExperimentDesign

__version__ = "0.0.1"
