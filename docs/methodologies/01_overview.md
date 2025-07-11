# GeoFleX Methodologies

At its core, a GeoFleX **methodology** is a self-contained set of rules that dictates how a geo-experiment is designed and analyzed. It provides the logic for the two most critical phases of an experiment:

1.  **Geo Assignment**: How to split your geographic locations into control and treatment groups.
2.  **Experiment Analysis**: How to calculate the treatment effect and statistical significance once the experiment has run.

By encapsulating these two components, GeoFleX allows for a flexible and extensible system where different statistical approaches can be compared and used interchangeably.

-----

## The `Methodology` Base Class

Every methodology in GeoFleX inherits from the `Methodology` base class. This class ensures a consistent interface by requiring each new methodology to implement a few key methods.

### Core Components of a Methodology

1.  **`_methodology_assign_geos`**: This is the heart of the design phase. This method contains the logic for assigning your geos into control and treatment groups. This could be a simple random assignment, a sophisticated matching algorithm based on historical data, or any other custom logic. It must return a `GeoAssignment` object.

2.  **`_methodology_analyze_experiment`**: This method defines how to analyze the results of a completed experiment. It takes the experiment's runtime data and the `ExperimentDesign` (which includes the geo assignment) and returns a pandas DataFrame with the calculated treatment effects, confidence intervals, and p-values for each metric.

3.  **`_methodology_is_eligible_for_design_and_data`**: Not all methodologies are suitable for every experimental design or dataset. This method acts as a guardrail, allowing the methodology to flag itself as ineligible for a given `ExperimentDesign`. For example, a methodology might not support multi-cell experiments (`n_cells > 2`) or designs that require pre-assigned geos. If this method returns `False`, GeoFleX will not use this methodology for that specific design.

-----

## Returning Intermediate Data for Deeper Analysis

Methodologies in GeoFleX can optionally return a second value: a dictionary called `intermediate_data`. This dictionary is a place to store any data, models, or artifacts generated during the assignment or analysis process that might be useful for debugging or deeper, more bespoke analysis.

For example, the built-in Geo-Based Regression (GBR) methodology returns the fitted linear model object, its parameters, and the covariance matrix. This allows advanced users to inspect the model's diagnostics, create custom plots, or perform further calculations that go beyond the standard summary results.

When you call `assign_geos` or `analyze_experiment`, you can set the `return_intermediate_data=True` argument to receive this dictionary alongside the main results.

-----

## Creating a Custom Methodology: An Example

Let's walk through creating a simple custom methodology: a basic Randomized Controlled Trial (RCT), where geos are assigned to treatment and control groups completely at random. We will also include an example of returning intermediate data.

We will create a new class, `RandomizedControlTrial`, that inherits from `geoflex.methodology.Methodology`.

```python
import geoflex as gx
import pandas as pd

@gx.register_methodology
class RandomizedControlTrial(gx.Methodology):
    """
    A simple Randomized Controlled Trial methodology.
    - Geos are assigned randomly to control and treatment groups.
    - Analysis is a simple comparison of means.
    """

    def _methodology_is_eligible_for_design_and_data(
        self,
        design: gx.ExperimentDesign,
        pretest_data: gx.GeoPerformanceDataset
    ) -> bool:
        """
        This methodology is eligible for any design, so we just return True.
        """
        return True

    def _methodology_assign_geos(
        self,
        experiment_design: gx.ExperimentDesign,
        historical_data: gx.GeoPerformanceDataset,
    ) -> tuple[gx.GeoAssignment, dict]:
        """
        Randomly assigns geos to control and treatment groups.
        """
        rng = experiment_design.get_rng() # Use the design's random number generator
        all_geos = list(historical_data.geos)
        rng.shuffle(all_geos)

        # Simple 50/50 split for a 2-cell experiment
        n_control = len(all_geos) // 2
        control_geos = set(all_geos[:n_control])
        treatment_geos = set(all_geos[n_control:])

        geo_assignment = gx.GeoAssignment(
            control=control_geos,
            treatment=[treatment_geos] # List of sets for treatment groups
        )

        # Store the shuffled list of geos as intermediate data
        intermediate_data = {"shuffled_geo_order": all_geos}

        return geo_assignment, intermediate_data

    def _methodology_analyze_experiment(
        self,
        runtime_data: gx.GeoPerformanceDataset,
        experiment_design: gx.ExperimentDesign,
        experiment_start_date: pd.Timestamp,
        experiment_end_date: pd.Timestamp,
        pretest_period_end_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict]:
        """
        A placeholder for a simple analysis. A real implementation would
        calculate the actual effect sizes and statistical significance.
        """
        # In a real scenario, you would implement your analysis logic here.
        # For this example, we'll return a dummy DataFrame and intermediate data.
        results = []
        for metric in [experiment_design.primary_metric] + experiment_design.secondary_metrics:
            results.append({
                "metric": metric.name,
                "cell": 1,
                "point_estimate": 100.0,
                "lower_bound": 50.0,
                "upper_bound": 150.0,
                "point_estimate_relative": 0.1,
                "lower_bound_relative": 0.05,
                "upper_bound_relative": 0.15,
                "p_value": 0.02
            })

        intermediate_data = {"analysis_timestamp": pd.Timestamp.now()}

        return pd.DataFrame(results), intermediate_data

```

### Registering Your Methodology

The final step is to make GeoFleX aware of your new methodology. This is done with the `@register_methodology` decorator, which you can see in the example above.

Once your class is decorated, GeoFleX can discover and use it just like any of the built-in methodologies. You can now specify `"RandomizedControlTrial"` as the `methodology` in your `ExperimentDesign` or `ExperimentDesignExplorationSpec`.

This powerful feature allows you to integrate your own proprietary models or statistical techniques directly into the GeoFleX framework, leveraging its evaluation and exploration capabilities for your custom methods.