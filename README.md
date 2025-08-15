# GeoFleX: A flexible and unified geo-experiment solution

**Disclaimer: This is not an official Google product.**

## What is GeoFleX?

GeoFleX is an open-source Python library designed to simplify and standardize the process of geo-experimentation. Geo-based experiments (or geo-tests) are a powerful tool for measuring the impact of marketing campaigns, product launches, and other market-level interventions. However, designing and analyzing these experiments can be complex, with many different methodologies to choose from.

GeoFleX helps users to:

1. **Design** a geo-experiment, by selecting an appropriate methodology and splitting geos into control and treatment groups.
2. **Evaluate** the design of a geo-experiment.
3. **Analyse** the results of geo experiment.

Its key features include:

- **Standardized Interface:** GeoFleX offers a consistent API for a wide range of geo-testing methodologies. This allows you to easily switch between different approaches (like Geo-Based Regression or Synthetic Controls) to find the one that best suits your data and experimental goals.

- **Novel Evaluation Framework:** At the core of GeoFleX is a powerful, non-parametric evaluation engine. It uses a custom time-series bootstrapping approach to conduct power analysis for any given methodology. This allows you to estimate the Minimum Detectable Effect (MDE) and understand the statistical power of your chosen design before you run the experiment. Additionally, since it is non-parameteric, it provides a fair like-for-like comparison between different methodologies, allowing you to easily select the best methodology for your geo experiment.

- **Robustness Checks:** The evaluation framework performs a variety of robustness checks based on the bootstrap samples, giving you confidence in the validity of your experimental design and the reliability of its results.

Whether you're a seasoned data scientist or new to geo-experimentation, GeoFleX provides the tools you need to design robust experiments and make data-driven decisions with confidence.

## Available Methodologies

GeoFleX allows you to choose from any of the following methodologies, or you can allow GeoFleX to automatically select the best one for your situation based on the data you provide:

1. Time Based Regression with Matched Markets (TBR-MM). [Paper](https://research.google/pubs/a-time-based-regression-matched-markets-approach-for-designing-geo-experiments/).
2. Time Based Regression with Random Assignment (TBR). [Paper](https://research.google/pubs/estimating-ad-effectiveness-using-geo-experiments-in-a-time-based-regression-framework/).
3. [COMING SOON] Trimmed Match (TM). [Paper](https://research.google/pubs/trimmed-match-design-for-randomized-paired-geo-experiments/).
4. Geo Based Regression (GBR). [Paper](https://research.google/pubs/measuring-ad-effectiveness-using-geo-experiments/).
5. [COMING SOON] Synthetic Controls.

## Getting Started

Check out the documentation and the quick start guides for getting started with either the UI or the python library.

## Citing GeoFleX

To cite this repository:

```
@software{geoflex_github,
  author = {Sam Bailey, Wincy Liu, Omri Goldstein, Sergei Dorogin, Daniel Kandel},
  title = {GeoFleX: A unified and flexible geo experimenation solution.},
  url = {https://github.com/google-marketing-solutions/geoflex},
  version = {0.0.3},
  year = {2025},
}
```