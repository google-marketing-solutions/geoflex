# Quick Start Guide

## Python Library Quick Start

This guide will walk you through an end-to-end example, from loading your data to analyzing the results of your experiment.

### 1. Installation
First, make sure you have GeoFleX installed. You can install with pip using the command below:

```bash
pip install "git+https://github.com/google-marketing-solutions/geoflex#egg=geoflex&subdirectory=lib"
```

### 2. Loading Your Data
The first step in any analysis is to load your historical performance data. This data should be at the geo-day level, meaning each row represents the performance of a single geographic area for a single day.

The primary data structure for handling this in GeoFleX is the GeoPerformanceDataset.

```python
import pandas as pd
import geoflex as gx

# Create a sample DataFrame with your historical data
# In a real scenario, you would load this from a file (e.g., CSV)
data = pd.DataFrame({
    "geo_id": [f"geo_{i}" for i in range(10) for _ in range(100)],
    "date": pd.date_range(start="2023-01-01", periods=100).tolist() * 10,
    "revenue": [1000 + i * 100 + (j * 10) + abs(50 * (i-5)) * abs(50 * (j-50)) for i in range(10) for j in range(100)],
    "cost": [100 + i * 10 + (j * 1) + abs(5 * (i-5)) * abs(50 * (j-50)) for i in range(10) for j in range(100)],
    "conversions": [50 + i * 5 + (j * 0.5) + abs(2 * (i-5)) * abs(50 * (j-50)) for i in range(10) for j in range(100)],
})

# Create a GeoPerformanceDataset object
historical_data = gx.GeoPerformanceDataset(data=data)

print("Successfully loaded historical data.")
print(f"Found {len(historical_data.geos)} geos and data from {historical_data.dates[0].date()} to {historical_data.dates[-1].date()}.")
```

### 3. Designing the Experiment

Next, you need to define your experiment's parameters using the ExperimentDesign class. This is where you specify your metrics, budget, experiment duration, and the methodology you want to use.

```python
# Define the experiment design
experiment_design = gx.ExperimentDesign(
    primary_metric="revenue",
    secondary_metrics=["conversions", gx.metrics.iROAS()],
    experiment_budget=gx.ExperimentBudget(
        value=-0.5,  # A 50% reduction in spend
        budget_type=gx.ExperimentBudgetType.PERCENTAGE_CHANGE,
    ),
    methodology="GBR",  # Geo-Based Regression
    runtime_weeks=4,
    n_cells=2 # 1 control, 1 treatment
)

print("Experiment design created:")
experiment_design.print_summary()
```

### 4. Assigning Geos

With your design defined, the next step is to assign your geographic locations to control and treatment groups. The assign_geos function uses the methodology specified in your ExperimentDesign to create an optimal split.

```python
# Assign geos to control and treatment groups
gx.assign_geos(experiment_design, historical_data)

print("\nGeo assignment complete. Updated design summary:")
experiment_design.print_summary()
```

### 5. Evaluating the Design (Power Analysis)

This is a key feature of GeoFleX. Before running your experiment, you can evaluate your design to understand its statistical properties. The ExperimentDesignEvaluator uses bootstrapping to simulate thousands of experiments based on your historical data. This allows it to calculate the standard error and, most importantly, the Minimum Detectable Effect (MDE) for your primary metric.

The MDE tells you the smallest effect size your experiment will be able to reliably detect with the desired statistical power (typically 80%).

```python
# Initialize the evaluator with your historical data
evaluator = gx.ExperimentDesignEvaluator(historical_data=historical_data)

# Evaluate the design
# This will run simulations to perform a power analysis
evaluator.evaluate_design(
    design=experiment_design,
    n_aa_simulations=100, # Number of A/A simulations for standard error estimation
    n_ab_simulations=100  # Number of A/B simulations for power validation
)

print("\nDesign evaluation complete. Final summary:")
experiment_design.print_summary(use_relative_effects_where_possible=True)
```

### 6. Saving Your Chosen Design

The experiment design can be serialized to a json string so you can save it for re-use later. This is important because
after you have designed your experiment, you will need to save the design somewhere so that at the end of the experiment when
you come back to analyse the results, you can re-load the experiment design and use it for the analysis step.

```python
# Convert the design to a json string
json_string = experiment_design.model_dump_json()

# Now you can save this json string somewhere.

# If you want to realod the design, you just need to load the json string back
# and then do:
reloaded_design = gx.ExperimentDesign.model_validate_json(json_string)
```

### 7. Analyzing the Experiment Results

After your experiment has run its course, you'll have a new dataset containing the performance data from the experiment period. You can use GeoFleX to analyze these results.

For this quick start, we'll simulate some post-experiment data.

```python
# Simulate the experiment to get runtime data
# We'll simulate a 5% lift in revenue for the treatment group
runtime_data = historical_data.simulate_experiment(
    experiment_start_date=pd.to_datetime("2023-04-11"),
    design=experiment_design,
    treatment_effect_sizes=[
        experiment_design.evaluation_results.get_mde(relative=False)["revenue"] * 0.5
    ] # Effect is 50% of the MDE
)


# Analyze the results
analysis_results = gx.analyze_experiment(
    experiment_design=experiment_design,
    runtime_data=runtime_data,
    experiment_start_date="2023-04-11",
)

print("\nExperiment analysis complete.")
```

### 8. Visualizing the Results

Finally, you can use the built-in visualization tools to display the analysis results in a clear and easy-to-read format.

```python
# Display the formatted analysis results
gx.display_analysis_results(
    analysis_results=analysis_results,
    alpha=experiment_design.alpha
)
```

This styled output will highlight statistically significant results and make it easy to interpret the outcome of your experiment.

## UI Quick Start

### Installing the UI

TODO: Add

### Using the UI

TODO: Add