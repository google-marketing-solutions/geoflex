# Analysing an experiment with GeoFleX

Once your geo-experiment has concluded, the final step is to analyze the collected data to measure the impact of your intervention. GeoFleX provides a straightforward analysis workflow that uses the same `ExperimentDesign` object you created for the design and evaluation phases.

This guide will walk you through loading your experiment data, running the analysis, and interpreting the results.

## The `analyze_experiment` function

The core of the analysis is the `gx.analyze_experiment` function. It takes your completed experiment design and the data collected during the experiment's runtime to calculate the treatment effects.

Here are the key parameters:

-   `experiment_design`: Your `ExperimentDesign` object. It's crucial that this object contains the final geo assignment that was used for the live experiment.
-   `runtime_data`: A `GeoPerformanceDataset` object containing the performance data from the entire experiment period, plus the historical data from before the start of the experiment.
-   `experiment_start_date`: A string (`YYYY-MM-DD`) indicating the first day of the treatment period.

For more advanced analyses, you can also use:

-   `experiment_end_date`: By default, the analysis period is determined by the `runtime_weeks` in your design. You can override this to analyze a shorter or longer period (e.g., to include a cool-down period).
-   `pretest_period_end_date`: If your experiment included a "washout" period (a gap between the pre-test data and the treatment start), you can specify the end of the pre-test data here. This is useful if there is a period of time where you were setting up your new treatment campaigns, which does not count as pre-test data because you have started making changes to the treatment geos, but is not stable yet so you want to exclude it from your analysis.

## Understanding the Results

The `analyze_experiment` function returns a pandas DataFrame where each row corresponds to a specific metric in a specific treatment cell. The key columns in the output are:

-   `metric`: The name of the metric being analyzed.
-   `cell`: The treatment cell number (1 for the first treatment group, 2 for the second, and so on).
-   `point_estimate`: The model's best estimate of the total absolute impact for that metric (e.g., an increase of $150,000 in revenue).
-   `lower_bound` & `upper_bound`: The lower and upper bounds of the confidence interval for the absolute impact.
-   `point_estimate_relative`: The point estimate expressed as a percentage change from the baseline (e.g., +5.2%). This is not calculated for ratio metrics like iROAS or CPiA.
-   `lower_bound_relative` & `upper_bound_relative`: The confidence interval for the relative impact.
-   `p_value`: The p-value associated with the hypothesis test.
-   `is_significant`: A boolean flag that is `True` if the result is statistically significant at the `alpha` level defined in your `ExperimentDesign`.

## Putting It All Together: An Example

Let's walk through a complete analysis example. We'll assume you have an `ExperimentDesign` from your planning phase and have just finished collecting your experiment data.

```python
import pandas as pd
import geoflex as gx

# --- Setup: Assume you have an experiment_design from the design phase ---
# This would typically be loaded from a file or stored from a previous step.
design_as_json = ...
experiment_design = gx.ExperimentDesign.model_validate_json(design_as_json)


# --- Analysis Starts Here ---

# 1. Load your experiment data
raw_runtime_data = ... # Load the experiment data from a file as a pandas dataframe
runtime_data = geoflex.GeoPerformanceDataset(
    data=raw_runtime_data
    geo_id_column="GMA",
    date_column="date"
)

# 2. Analyze the results
# Use the analyze_experiment function with your design and runtime data.
analysis_results = gx.analyze_experiment(
    experiment_design=experiment_design,
    runtime_data=runtime_data,
    experiment_start_date="2023-04-11",
)

# 3. Display the results
# Use the display_analysis_results helper function for a clean, formatted output.
# It highlights significant results and formats confidence intervals.
gx.display_analysis_results(
    analysis_results=analysis_results,
    alpha=experiment_design.alpha
)
```

The output will be a styled table, making it easy to see which metrics had a statistically significant impact. Positive significant results are highlighted in green, and negative ones in pink.

## Deep Dive Analysis

Some methodologies, like Geo-Based Regression (GBR), can produce additional diagnostic plots to help you understand the model's behavior. To generate these, you can set `with_deep_dive_plots=True` when calling `analyze_experiment`.

```python
# This will generate and display additional plots for the GBR methodology.
analysis_results = gx.analyze_experiment(
    experiment_design=experiment_design,
    runtime_data=runtime_data,
    experiment_start_date=experiment_start_date,
    with_deep_dive_plots=True
)
```

This can provide valuable insights into how the model is performing and increase confidence in the results.