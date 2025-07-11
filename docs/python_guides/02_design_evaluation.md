# Guide: Evaluating Experiment Designs

After defining your `ExperimentDesign`, the next crucial step is to evaluate it. This pre-experiment analysis is a cornerstone of the GeoFleX library, allowing you to understand the statistical power and robustness of your setup before committing to a live test. This guide will walk you through the methodology behind GeoFleX's evaluation engine and demonstrate how to use it.

## The GeoFleX Evaluation Engine

At the heart of GeoFleX's evaluation capabilities is the `ExperimentDesignEvaluator`. This powerful tool uses a non-parametric, time-series bootstrapping approach to simulate your experiment thousands of times based on your historical data. This process allows for a robust, data-driven assessment of how your chosen methodology will perform.

The evaluation is primarily based on two types of simulations:

  * **A/A Simulations**: These simulations assume a treatment effect of zero. By running many of these "null" experiments, the evaluator can accurately estimate the variance and standard error of your chosen metrics under the null hypothesis. This is the foundation for calculating the Minimum Detectable Effect (MDE).
  * **A/B Simulations**: In these simulations, a synthetic treatment effect is injected into the data. The size of this effect is determined by the MDE calculated from the A/A simulations. By observing how often the methodology correctly detects this synthetic effect, the evaluator can empirically validate the statistical power of the design.

## Minimum Detectable Effect (MDE)

The MDE is the smallest true effect that your experiment can reliably detect with a given level of statistical power (typically 80%). A lower MDE indicates a more powerful and sensitive experiment design.

GeoFleX calculates the MDE for each metric in your design using the standard error derived from the A/A simulations. The formula combines this standard error with the desired alpha (significance level) and power:

`MDE = standard_error * (z_alpha + z_power)`

Where:

  * `z_alpha` is the critical value from the normal distribution for your chosen `alpha`.
  * `z_power` is the critical value from the normal distribution for your desired power.

### A Note on Cost-per-Metric Metrics

For metrics that are defined as a cost per something (e.g., CPiA, or Cost Per Incremental Acquisition), the MDE calculation is slightly different. Since the incremental effect is in the denominator, a standard MDE is not meaningful. Instead, GeoFleX inverts the metric (to acquisition per cost) for the power calculation and then re-inverts the result. In this scenario, the MDE should be interpreted as the **Maximum Detectable Effect**. A higher value is better, as it indicates the ability to detect smaller, more desirable changes in cost efficiency.

## Robustness Checks

The bootstrapping process enables a series of rigorous validation checks to ensure your experiment design is sound. If the number of bootstrap samples is too low for these checks to be conclusive, GeoFleX will issue a warning.

Key robustness checks include:

  * **Coverage**: This check verifies that the confidence intervals produced by the methodology are reliable. For a 90% confidence interval, we expect the true effect (which is zero in an A/A test) to fall within the calculated interval 90% of the time. The evaluator flags designs where the coverage is significantly lower than expected.
  * **Bias**: The evaluator checks if the effect estimates are systematically biased. In an A/A simulation, the average estimated effect should be centered around zero. Any significant deviation indicates a potential bias in the methodology for your specific dataset.
  * **Power**: Using the A/B simulations, the evaluator confirms that the empirical power of the test matches the target power (usually 80%). If the methodology detects the injected synthetic effect less often than expected, it suggests the design may be underpowered.

## Representativeness Scorer

When the `effect_scope` of your `ExperimentDesign` is set to `'all_geos'`, it is critical that your treatment group is a good representation of your entire market. GeoFleX includes a `GeoAssignmentRepresentativenessScorer` to quantify this.

The scorer works by first calculating a similarity matrix based on the historical time-series data of your specified metrics. It computes the cosine similarity between every pair of geos. A score of 1 indicates perfect similarity, while -1 indicates perfect opposition.

When an assignment is evaluated, the scorer calculates the average similarity between each geo in the entire population and its closest corresponding geo in the treatment group. A high score (close to 1.0) indicates that the treatment group is highly representative of the overall market, giving you confidence that your results can be generalized.

## How to Evaluate a Design

Evaluating a design is straightforward. First, you initialize the `ExperimentDesignEvaluator` with your `GeoPerformanceDataset`. Then, you call the `evaluate_design` method, passing your `ExperimentDesign` object.

```python
import geoflex as gx
import pandas as pd

# Assume 'historical_data' and 'experiment_design' are already created
# (See the "Creating an Experiment Design" guide for details)

# 1. Initialize the evaluator with your historical data
evaluator = gx.ExperimentDesignEvaluator(historical_data=historical_data)

# 2. Evaluate the design
# GeoFleX can automatically determine the required number of simulations.
# Alternatively, you can specify them with n_aa_simulations and n_ab_simulations.
evaluator.evaluate_design(
    design=experiment_design
)

# 3. Print the summary to see the results
# The MDE and other evaluation metrics will now be populated.
print("\nDesign evaluation complete. Final summary:")
experiment_design.print_summary()

```

After running the evaluation, the `evaluation_results` attribute of your `experiment_design` object will be populated. The `print_summary()` method will now include the MDE for your primary and secondary metrics, the representativeness score (if applicable), and a summary of any failing checks, giving you a complete picture of your design's quality.

## The Bootstrapper: Powering the Evaluation

The reliability of the evaluation process hinges on its ability to generate realistic, simulated datasets that preserve the complex temporal patterns and correlations of the original historical data. This is the role of the `MultivariateTimeseriesBootstrap` class. It is a sophisticated engine designed to create new time-series samples that are statistically similar to the real-world data you provide.

### How the Bootstrapper Works

Instead of simple random sampling, which would destroy the time-dependent structure of the data, the bootstrapper uses a more advanced block-based method. Here’s a step-by-step breakdown of the process:

1.  **Trend and Noise Decomposition**: The first step is to separate the underlying signal from the noise in your time-series data. GeoFleX uses a technique called Seasonal-Trend-Loess (STL) decomposition for this. It breaks down each geo's time series into three components: a long-term trend, a recurring seasonal pattern (e.g., weekly), and a remainder, which we consider to be the noise. The model can be either additive (`y = trend + noise`) or multiplicative (`y = trend * noise`), with the multiplicative option ensuring that the bootstrapped data remains non-negative, which is typical for performance metrics.

2.  **Differencing the Trend**: The long-term trend component is "differenced," meaning we look at the day-over-day changes rather than the absolute values. This helps to make the trend component stationary, which is a key requirement for effective bootstrapping.

3.  **Block-Based Sampling**: The core of the methodology is the block bootstrap. The noise and the differenced trend data are sliced into sequential, non-overlapping blocks of a fixed size (e.g., 4 weeks of daily data). These blocks, which preserve the short-term auto-correlation within the time series, are then shuffled and stitched back together to create a new, full-length time series. This shuffling can be done in two ways:
    * **Permutation**: Each block from the original data is used exactly once in the new sample, but in a random order.
    * **Random Sampling**: Blocks are sampled with replacement, meaning some blocks from the original data may appear multiple times in the new sample, while others may not appear at all.

4.  **Reconstructing the Time Series**: Once the blocks are shuffled, the process is reversed. The differenced trend is cumulatively summed (integrated) to reconstruct a new trend line. This new trend is then combined with the shuffled noise component (either by adding or multiplying, depending on the model type) to create a complete, new bootstrapped sample of your historical data.

This process is repeated to generate many unique, statistically plausible variations of your historical data.

### The Role of Bootstrapping in Evaluation

The `ExperimentDesignEvaluator` leverages these bootstrapped samples to conduct its A/A and A/B simulations. Here’s how it fits together:

1.  **Generating a Sample**: For each simulation run, the evaluator requests a new bootstrapped dataframe from the `MultivariateTimeseriesBootstrap`. This sample serves as the "ground truth" for a single simulated experiment.

2.  **Assigning Geos and Running the Simulation**: For each bootstrapped sample, the evaluator runs the `assign_geos` method of your chosen methodology to create a new control/treatment split. It then simulates the experiment on this sample.
    * In an **A/A simulation**, no synthetic effect is added. The analysis is run to see what kind of random noise and variance the methodology produces when there is no true effect. The standard error of the point estimates across all these A/A runs gives us the standard error for the MDE calculation.
    * In an **A/B simulation**, a synthetic effect (equal to the calculated MDE) is injected into the treatment group's data for the primary metric.

3.  **Aggregating Results**: The results from thousands of these simulated experiments are collected. By analyzing the distribution of outcomes from the A/A tests, the evaluator can verify the robustness checks for bias and coverage. By analyzing the results from the A/B tests, it can calculate the empirical power—the percentage of simulations where the injected effect was correctly identified as statistically significant.

By using this bootstrapping approach, GeoFleX provides a robust and non-parametric way to evaluate any experiment design, giving you a clear and reliable estimate of its power and validity before you use it in the real world, and a fair like-for-like comparison between different methodologies.