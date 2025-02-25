# GeoFleX: A flexible and unified geo-experiment solution

**Disclaimer: This is not an official Google product.**

## Introduction

TODO: Add general intro

## GeoFleX Python Package

GeoFleX can be used as a unified python package, from which a range of geo
experiment methodologies and designs can be evaluated and used. It provides a
standard interface to:

1. **Design** a geo-experiment.
2. **Evaluate** the design of a geo-experiment.
3. **Analyse** a geo experiment.

You can choose from any of the following methodologies, or allow GeoFleX to
automatically select the best one for your situation based on the data you
provide:

1. Time Based Regression with Matched Markets (TBR-MM). [Paper](https://research.google/pubs/a-time-based-regression-matched-markets-approach-for-designing-geo-experiments/).
2. Time Based Regression with Random Assignment (TBR). [Paper](https://research.google/pubs/estimating-ad-effectiveness-using-geo-experiments-in-a-time-based-regression-framework/).
3. Trimmed Match (TM). [Paper](https://research.google/pubs/trimmed-match-design-for-randomized-paired-geo-experiments/).
4. Geo Based Regression (GBR). [Paper](https://research.google/pubs/measuring-ad-effectiveness-using-geo-experiments/).

GeoFleX will automatically select the best design and methodology by treating
it as a hyperparameter optimization problem. It uses google-vizier behind the
scenes to select the best design to maximize the power on the primary metric.