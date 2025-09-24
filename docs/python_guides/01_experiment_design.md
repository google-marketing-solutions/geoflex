# Defining an Experiment Design with GeoFleX

At the heart of any geo-experiment in GeoFleX is the `ExperimentDesign` object. This class acts as a comprehensive blueprint, defining all the essential parameters of your test, from the metrics you want to measure to the methodology used for the analysis.

A minimal `ExperimentDesign` requires just three things: a primary metric, a methodology, and the experiment's duration.

```python
import geoflex as gx

# A minimal experiment design
design = gx.ExperimentDesign(
    primary_metric="revenue",
    methodology="GBR",
    runtime_weeks=4
)

design.print_summary()
```

This creates a simple 2-cell (treatment vs. control) A/B test design using the Geo-Based Regression methodology, set to run for 4 weeks.

While this is a valid start, a real-world experiment often requires more detailed configuration. Let's explore the key parameters of the `ExperimentDesign` class in detail.

## Key Parameters of ExperimentDesign

### `primary_metric` and `secondary_metrics`

Metrics are the key performance indicators you want to measure. GeoFleX requires you to specify at least one primary_metric, which should be the main outcome you're using to make a decision. You can also provide a list of secondary_metrics to track other important outcomes.

A metric can be a simple string referring to a column in your data, or a more complex `Metric` object for cost-based calculations.

- `name`: The display name of the metric (e.g., "Revenue", "iROAS").

- `column`: The corresponding column name in your GeoPerformanceDataset. If not provided, it defaults to the name.

- `metric_per_cost`: Boolean. If True, the metric is a "metric per cost" ratio (e.g., ROAS).

- `cost_per_metric`: Boolean. If True, the metric is a "cost per metric" ratio (e.g., CPA).

- `cost_column`: The column name for the cost data, required if metric_per_cost or cost_per_metric is True.

An example of creating a custom cost-based metric is shown below:

```python
# Inverse iroas is 1/iroas, or the incremental cost divided by the incremental revenue.

inverse_iroas = gx.metrics.Metric(
    name="Inverse iROAS",
    column="revenue",
    cost_column="cost",
    cost_per_metric=True
)
```

GeoFleX provides two built-in cost-based metrics: `iROAS` and `CPiA`.

- **iROAS (Incremental Return On Ad Spend)**: Measures the incremental revenue generated for each incremental dollar of ad spend.
  - Formula: $iROAS=\frac{revenue_{treatment} - revenue_{control}}{cost_{treatment} - cost_{control}}$
​`
  - Usage: `gx.metrics.iROAS(return_column="revenue", cost_column="cost")`

- **CPiA (Cost Per Incremental Acquisition)**: Measures the incremental cost for each incremental acquisition (or conversion).
  - Formula: $CPiA=\frac{cost_{treatment} - cost_{control}}{ conversions_{treatment} - conversions_{control}}$
  - Usage: `gx.metrics.CPiA(conversions_column="conversions", cost_column="cost")`


```python
# Example with multiple metrics, including iROAS
design = gx.ExperimentDesign(
    primary_metric="revenue",
    secondary_metrics=[
        "conversions",
        gx.metrics.iROAS(return_column="revenue", cost_column="cost")
    ],
    methodology="GBR",
    runtime_weeks=4,
    # A budget is required for cost-based metrics like iROAS
    experiment_budget=gx.ExperimentBudget(
        value=-0.5, # 50% spend decrease
        budget_type=gx.ExperimentBudgetType.PERCENTAGE_CHANGE
    )
)
```

### `experiment_budget`

The experiment_budget parameter is crucial for defining the nature of the intervention in the treatment groups. It is encapsulated in the ExperimentBudget object, which has two main attributes: value and budget_type.

If you are running a standard A/B test with no change in marketing spend, you can omit this parameter. It will default to a 0% change. However, if you include any cost-based metrics like iROAS or CPiA, a non-zero budget change must be specified.

The budget_type can be one of three values from the `ExperimentBudgetType` enum:

- `PERCENTAGE_CHANGE`: The change in spend as a percentage of the Business-As-Usual (BAU) spend.

- `DAILY_BUDGET`: A fixed daily incremental budget.

- `TOTAL_BUDGET`: A fixed total incremental budget for the entire experiment duration.

Here is how you can model common geo-testing scenarios:

#### Go Dark Test

A "go dark" test involves completely cutting off spend in the treatment geos to measure the baseline contribution of the marketing channel. This is modeled as a 100% decrease in spend.

```python
# Go Dark: -100% spend change
go_dark_budget = gx.ExperimentBudget(
    value=-1.0,
    budget_type=gx.ExperimentBudgetType.PERCENTAGE_CHANGE
)
```

#### Heavy Up Test

A "heavy up" test involves increasing the marketing spend in the treatment geos. The budget you specify is the incremental amount on top of the existing BAU spend.

```python
# Heavy Up: Increase spend by a total of $500,000 over the experiment
heavy_up_budget = gx.ExperimentBudget(
    value=500000,
    budget_type=gx.ExperimentBudgetType.TOTAL_BUDGET
)
```

#### Hold Back Test

A "hold back" test is used for launching a new campaign where there is no pre-existing BAU spend. The structure is identical to a heavy-up test, but the interpretation is different: the budget represents the total spend for the new activity.

```python
# Hold Back: Spend $10,000 per day on a new campaign
hold_back_budget = gx.ExperimentBudget(
    value=10000,
    budget_type=gx.ExperimentBudgetType.DAILY_BUDGET
)
```

#### A/B Test (No Spend Change)

This is the default scenario. If you don't specify an experiment_budget, GeoFleX assumes a 0% change in spend, which is appropriate for tests where the intervention is not a budget change (e.g., testing different ad creatives with the same budget).

```python
# A/B Test with no budget change
# Simply omit the experiment_budget parameter
design = gx.ExperimentDesign(
    primary_metric="revenue",
    methodology="GBR",
    runtime_weeks=4
)
```

### `methodology` and `methodology_parameters`

This string parameter specifies which statistical model GeoFleX should use for the analysis. You can see a list of available methodologies with `gx.list_methodologies()`. Some methodologies have tunable parameters, which you can specify in the `methodology_parameters` dictionary.

```python
# Using Geo-Based Regression (GBR) with a specific linear model type
design = gx.ExperimentDesign(
    primary_metric="revenue",
    methodology="GBR",
    methodology_parameters={"linear_model_type": "robust_ols"},
    runtime_weeks=4
)
```

It's common not to know which methodology might be best for your experiment, so you can easily try different methodologies and methodology parameters and select the best combination for your dataset using the explorer. This is explained in more detail in the "Selecting the best experiment design" guide.

### `runtime_weeks`

An integer specifying the duration of the experiment in weeks. You can also use the explorer here to try designs with lots of different runtimes, to select the best runtime for your experiment. This is explained in more detail in the "Selecting the best experiment design" guide.

### `n_cells`

The total number of groups in the experiment. For a standard test, `n_cells=2` (one control group, one treatment group). For a multi-cell experiment (e.g., Control vs. Treatment A vs. Treatment B), you would set `n_cells=3` or more.

### `alpha` and `alternative_hypothesis`

These parameters control the statistical hypothesis tests:

- `alpha`: The significance level, typically set to 0.1 (90% confidence) or 0.05 (95% confidence).

- `alternative_hypothesis`: Specifies the test direction. Can be 'two-sided', 'greater' (to test for a positive effect), or 'less' (to test for a negative effect). We typically recommend a `two-sided` test.

### `geo_eligibility`

This parameter allows you to enforce business rules or pre-determined assignments for your geos. It uses the `GeoEligibility` class to specify sets of geos that must be in the control group, one of the treatment groups, or excluded from the experiment altogether. Any geos not explicitly mentioned will be considered "flexible" and can be assigned to any non-excluded group by the chosen methodology.

In the below (contrived) example, New York and Boston are both forced to be in the control group, California and Oregon must be in the first treatment group, Texas can be in either of the two treatment groups, Florida must be either in the second treatment group or excluded, and Alaska and Hawaii must both be excluded.

```python
# Example of setting geo eligibility constraints
# Assume a 3-cell experiment (1 control, 2 treatment arms)
eligibility = gx.GeoEligibility(
    control={"New York", "Boston"},
    treatment=[
        {"California", "Oregon", "Texas"},  # Eligible for treatment arm 1
        {"Texas", "Florida"}      # Eligible for treatment arm 2
    ],
    exclude={"Alaska", "Hawaii", "Florida"}
)

design = gx.ExperimentDesign(
    primary_metric="revenue",
    methodology="GBR",
    runtime_weeks=4,
    n_cells=3,
    geo_eligibility=eligibility
)
```

### `cell_volume_constraint`
This parameter gives you fine-grained control over the composition of your experimental groups using the CellVolumeConstraint class. This is particularly useful for balancing groups to ensure a fair comparison.

You can constrain the groups based on:

- `MAX_GEOS`: The maximum number of locations allowed in each group.
- `MAX_PERCENTAGE_OF_METRIC`: The maximum share of a key metric (like revenue or cost) that each group can represent.

The values attribute should be a list with a length equal to n_cells, where the first value corresponds to the control group and the subsequent values correspond to the treatment groups. You can use `None` for any group you don't wish to constrain.

```python
# Example 1: Constraining by the number of geos
# Control can have any number of geos, treatment group is limited to 10.
max_geos_constraint = gx.CellVolumeConstraint(
    values=[None, 10],
    constraint_type=gx.CellVolumeConstraintType.MAX_GEOS
)

# Example 2: Constraining by percentage of a metric
# Control group can't exceed 40% of total pre-experiment revenue.
# First treatment group can't exceed 50%, and
# second treatment group is unconstarined.
max_rev_constraint = gx.CellVolumeConstraint(
    values=[0.4, 0.5, None],
    constraint_type=gx.CellVolumeConstraintType.MAX_PERCENTAGE_OF_METRIC,
    metric_column="revenue" # The metric to base the percentage on
)

design = gx.ExperimentDesign(
    primary_metric="revenue",
    methodology="GBR",
    runtime_weeks=4,
    cell_volume_constraint=max_rev_constraint
)
```

### `effect_scope`

This parameter ('all_geos' or 'treatment_geos') is crucial as it defines the scope of your experiment's conclusion. It tells GeoFleX what population you are trying to make an inference about.

- **`all_geos` (default)**: Use this when you want to understand the average effect of your intervention across your entire market. For the results to be generalizable, the treatment group must be a good miniature representation of all your geos. To ensure this, when effect_scope is set to 'all_geos', GeoFleX calculates a representativeness_score during the design evaluation. A high score indicates a good, representative match between the treatment group and the overall market.

- **`treatment_geos`**: Use this when your intervention is only relevant to a specific subset of your geos, and you are not trying to generalize the findings to the whole market. For example, you might be testing a new premium delivery service that is only available in large urban centers. In this case, you only care about the impact within those specific cities. Because you are not trying to extrapolate the results, the treatment group does not need to be representative of the entire geo population. Consequently, GeoFleX will not calculate a representativeness_score when this scope is selected, as it's not a relevant measure of design quality for this type of question.

### `random_seed`

An integer used to ensure that any random processes (like geo assignment) are reproducible.