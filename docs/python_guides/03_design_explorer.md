# Guide: Selecting the Best Experiment Design with the GeoFleX Explorer

Finding the optimal design for a geo-experiment can be a complex task. There are many interacting parameters to consider, from the choice of statistical methodology to the experiment's runtime. The GeoFleX Explorer simplifies this process by treating design selection as an optimization problem. It intelligently searches through a wide range of possible configurations to find the design that is best suited for your specific data and goals.

Under the hood, GeoFleX leverages the powerful optimization library [Optuna](https://optuna.org/) to efficiently navigate the vast search space of potential designs. The primary goal of the explorer is to find a design that minimizes the **Minimum Detectable Effect (MDE)** for your primary metric. A lower MDE means your experiment is more sensitive and can detect smaller changes, increasing its value.

If you are aiming to generalize your findings across your entire market (i.e., you've set `effect_scope='all_geos'`), the explorer will also simultaneously optimize for the highest **representativeness score**. This ensures that the selected treatment group is a good miniature version of your overall geo landscape, giving you confidence in the generalizability of your results.

## Defining the Exploration Space with `ExperimentDesignExplorationSpec`

Instead of creating a single `ExperimentDesign`, you define a search space using the `ExperimentDesignExplorationSpec` class. This class is very similar to `ExperimentDesign`, but it allows you to provide a list of candidate values for several key parameters. The explorer will then test different combinations of these values to find the optimal mix.

You don't need to redefine all the parameters that were covered in the "Creating an Experiment Design" guide. The key is to specify the parameters you want to explore by providing a list of candidates.

Here are the parameters you can explore over:

  * `runtime_weeks_candidates`: A list of possible experiment durations (in weeks).
  * `eligible_methodologies`: A list of methodology names (as strings) that you want to consider (e.g., `["GBR", "TBR", "TM"]`).
  * `methodology_parameter_candidates`: A dictionary to specify parameter candidates for different methodologies. The key is the methodology name, and the value is another dictionary where keys are parameter names and values are lists of candidate values.
  * `experiment_budget_candidates`: A list of different `ExperimentBudget` objects to test.
  * `cell_volume_constraint_candidates`: A list of different `CellVolumeConstraint` objects.
  * `geo_eligibility_candidates`: A list of different `GeoEligibility` configurations.
  * `random_seeds`: A list of random seeds to try for the geo assignment process, allowing you to explore the impact of different random splits.

<!-- end list -->

```python
import geoflex as gx

# Define an exploration specification
# This tells the explorer what combinations of parameters to test.
exploration_spec = gx.ExperimentDesignExplorationSpec(
    primary_metric="revenue",
    secondary_metrics=["conversions", gx.metrics.iROAS()],

    # --- Parameters to explore ---
    runtime_weeks_candidates=[4, 6, 8],
    eligible_methodologies=["GBR", "TBR", "TM"],
    experiment_budget_candidates=[
        gx.ExperimentBudget(value=-1.0, budget_type=gx.ExperimentBudgetType.PERCENTAGE_CHANGE),
        gx.ExperimentBudget(value=-0.5, budget_type=gx.ExperimentBudgetType.PERCENTAGE_CHANGE)
    ],
    methodology_parameter_candidates={
        "GBR": {
            "linear_model_type": ["wls", "robust_ols"]
        }
    },
    random_seeds=[0, 1, 2],

    # --- Fixed parameters ---
    n_cells=2,
    alpha=0.1,
    effect_scope=gx.EffectScope.ALL_GEOS
)
```

## Running the Exploration

Once your `ExperimentDesignExplorationSpec` is defined, you can initialize the `ExperimentDesignExplorer` and start the exploration process.

The `explore()` method is the main entry point. You need to specify the `max_trials`, which is the maximum number of valid experiment designs the explorer will evaluate.

```python
# Assume 'historical_data' and 'exploration_spec' are already defined

# 1. Initialize the explorer
explorer = gx.ExperimentDesignExplorer(
    historical_data=historical_data,
    explore_spec=exploration_spec
)

# 2. Run the exploration
# This will test various combinations and find the best designs.
explorer.explore(
    max_trials=100,
    n_jobs=-1 # Use all available CPU cores to speed up the process
)

print("Exploration complete!")
```

## A Practical Two-Stage Approach for Efficient Exploration

Running a full evaluation with a large number of simulations for every single design candidate can be computationally expensive and slow. A more practical and efficient workflow is a two-stage approach:

**Stage 1: Broad Exploration with Low Simulations**

First, explore a large number of potential designs using a small number of simulations. The goal here is to quickly identify a shortlist of promising candidates without getting bogged down in precise MDE calculations. Running with a low number of A/A simulations (e.g., 50) and zero A/B simulations is a good starting point.

```python
# Stage 1: Explore 100 designs with a low number of simulations
explorer.explore(
    max_trials=100,
    aa_simulations_per_trial=50,
    ab_simulations_per_trial=0, # Skip A/B tests for speed
    n_jobs=-1
)

# Get a summary of the top 5 designs from the initial exploration
top_5_designs_summary = explorer.get_design_summaries(top_n=5, style_output=True)
top_5_designs_summary
```

**Stage 2: Intensive Evaluation of Top Candidates**

After the initial exploration, you will have a ranked list of the best-performing designs based on the estimated MDE. Now, you can take the top few candidates (e.g., the top 5) and run a full, more intensive evaluation on them with a higher number of simulations. This will give you a more accurate and robust estimate of their MDE and validate their power.

The `extend_top_n_designs()` method makes this easy. It takes the top `n` designs and runs additional simulations, adding them to the existing results.

```python
# Stage 2: Run a full evaluation on the top 5 designs
# GeoFleX will automatically determine the sufficient number of simulations
# needed for robust validation checks.
explorer.extend_top_n_designs(
    top_n=5
)

# Get the final, fully-evaluated summary of the top 5 designs
final_top_5_summary = explorer.get_design_summaries(top_n=5, style_output=True)
final_top_5_summary
```

By following this two-stage process, you can efficiently navigate a vast landscape of design possibilities, saving time while still ensuring you select a powerful and robust design for your geo-experiment.

## Saving Your Chosen Design

Once the explorer has identified the optimal design for your experiment, it's crucial to save it. This saved design object is not just a record of your decisions; it's a required input for the final step in the workflow: analyzing the experiment's results.

GeoFleX makes it easy to persist your chosen design to a JSON file.

### Retrieving and Saving the Best Design

The `ExperimentDesignExplorer` keeps track of all the designs it has evaluated. You can easily retrieve the top-performing design and save it for later. This is important because you will need it again to analyse the results of the experiment.

```python
# Assume 'explorer' has completed its exploration run

# 1. Retrieve the best design from the explorer based on the design id
selected_design = explorer.get_design_by_id("00ca7e20-9d30-4b72-b4bf-da9866cd5764")

# 2.Convert the design to a json string
json_string = selected_design.model_dump_json()

# Now you can save this json string somewhere.

# If you want to realod the design, you just need to load the json string back
# and then do:
reloaded_design = gx.ExperimentDesign.model_validate_json(json_string)
```