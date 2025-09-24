# Python Guides for GeoFleX

Welcome to the Python guides for GeoFleX. These guides provide a comprehensive walkthrough of the entire geo-experimentation workflow, from designing your first experiment to analyzing the final results.

## Installation

To get started, install GeoFleX directly from the GitHub repository using pip:

```bash
pip install "git+https://github.com/google-marketing-solutions/geoflex#egg=geoflex&subdirectory=lib"
```

## The GeoFleX Workflow

The typical workflow in GeoFleX follows four main stages, each covered by a dedicated guide. We recommend following them in order to get a complete understanding of the library's capabilities.

1.  **[Creating an Experiment Design](/python_guides/01_experiment_design.md)**: Learn how to define the blueprint for your experiment using the `ExperimentDesign` class. This guide covers all the essential parameters, from setting up metrics and budgets to defining your statistical methodology.

2.  **[Evaluating Experiment Designs](/python_guides/02_design_evaluation.md)**: Before you launch your experiment, it's crucial to understand its statistical power. This guide walks you through using the `ExperimentDesignEvaluator` to run simulations of your experiment on historical data, calculate the Minimum Detectable Effect (MDE), and run robustness checks.

3.  **[Selecting the Best Experiment Design](/python_guides/03_design_explorer.md)**: Finding the optimal set of parameters for your experiment can be challenging. This guide introduces the `ExperimentDesignExplorer`, a powerful tool that automates the search for the most powerful and robust design for your specific needs.

4.  **[Analysing an Experiment](/python_guides/04_experiment_analysis.md)**: Once your experiment is complete, this guide shows you how to use the `analyze_experiment` function to calculate the treatment effect, determine statistical significance, and interpret the results.
