# Time-Based Regression (TBR)

The **Time-Based Regression (TBR)** methodology is a robust statistical technique for analyzing geo-experiments by leveraging time series data. It creates a model of the relationship between the control and treatment groups during a pre-test period and uses this model to forecast the "counterfactual"—what would have happened in the treatment group had they not received the treatment. The incremental impact is then calculated as the difference between this forecasted counterfactual and the actual observed performance.

The implementation of TBR in GeoFleX acts as a wrapper around the [`matched_markets`](https://github.com/google/matched_markets) library, leveraging its robust and battle-tested implementation of the methodology.

For a more detailed explanation of the methodology, see the original paper: [Estimating Ad Effectiveness Using Geo Experiments in a Time-Based Regression Framework (Kerman, Wang and Vaver, 2017)](https://research.google/pubs/estimating-ad-effectiveness-using-geo-experiments-in-a-time-based-regression-framework/).

---
## How TBR Works

TBR is based on creating a linear model that predicts the performance of the treatment group based on the performance of the control group over time.

### Geo Assignment

In the design phase, GeoFleX's TBR implementation assigns geos to control and treatment groups **randomly**. This helps ensure that, on average, the two groups are comparable before the experiment begins. The assignment process also supports stratification based on a metric to ensure the groups are balanced.

### Analysis with a Time-Series Model

After the experiment, TBR analyzes the results using a linear regression model that is fit on the pre-test data:

$Y_{t} = \beta_{0} + \beta_{1}X_{t} + \epsilon_{t}$

Where:
* $Y_{t}$ is the total value of the response metric (e.g., revenue) for the **treatment group** at time *t*.
* $X_{t}$ is the total value of the response metric for the **control group** at time *t*.

Once this model is trained on the pre-test data, it is used to predict the counterfactual performance of the treatment group during the test period, using the actual performance of the control group as input. The difference between the actual, observed values in the treatment group and these predicted values gives us the estimated treatment effect.

Before finalizing the analysis, the GeoFleX implementation of TBR runs a series of **pre-analysis diagnostics** to ensure the data is high quality. These diagnostics check for a strong correlation between the control and treatment groups in the pre-test period, identify and remove outlier dates, and flag any other data quality issues that might impact the validity of the model.

### iROAS Analysis

The TBR methodology in GeoFleX also supports Incremental Return on Ad Spend (iROAS) analysis. When a cost column is provided for a metric, it uses a specialized `TBRiROAS` model to directly estimate the return on investment from the advertising spend change.

The iROAS calculation follows a two-stage process:

1.  **Calculate Incremental Response and Cost**: First, two separate TBR models are trained on the pre-test data: one for the response metric (e.g., revenue) and one for the cost metric. These models are then used to predict the counterfactual for both response and cost in the treatment group during the test period. The incremental values are the differences between the actuals and these predictions:
    *   `Incremental Response = Actual Response - Predicted Response`
    *   `Incremental Cost = Actual Cost - Predicted Cost`

2.  **Estimate iROAS with a Second Model**: A final linear regression model is fit to estimate the relationship between these incremental values, with the incremental response as the dependent variable and incremental cost as the independent variable. The model is forced through the origin (i.e., no intercept is fitted), which reflects the assumption that zero incremental cost should lead to zero incremental response.

The coefficient of this final model, $\beta_{iROAS}$, is the estimated iROAS:

`Incremental Response` = $\beta_{iROAS}}$ * `Incremental Cost`

This two-stage approach allows for a robust estimation of the return on ad spend by modeling the underlying time-series dynamics of both response and cost independently before calculating their relationship.

### Comparison with Synthetic Controls

TBR can be seen as a simplified version of the Synthetic Controls methodology. While Synthetic Controls finds an optimal set of weights for the control group geos to best match the treatment group's pre-test performance, TBR effectively treats all control geos as a single unit. In essence, TBR is equivalent to a Synthetic Control model where every geo in the control group is given an equal weight.

This makes TBR a good choice when the control and treatment groups are already well-matched and have a strong, stable relationship over time. However, if the relationship is more complex or if there isn't a single, simple control group that tracks the treatment group well, Synthetic Controls offers a more flexible and powerful approach by creating a more nuanced counterfactual.

---
## Restrictions and Best Practices

While TBR is a powerful methodology, there are several key considerations to keep in mind:

*   **Two-Cell Designs Only**: The TBR implementation in GeoFleX is designed for simple A/B tests and only supports experiments with two cells (one control group and one treatment group).
*   **Random Assignment is Required**: The TBR methodology in GeoFleX requires that geos be assigned randomly. It **cannot** handle experiments where specific geos are forced into the control or treatment groups.
*   **No Inverse Cost Metrics**: TBR does not support metrics that are inverse cost functions, such as Cost Per Acquisition (CPA). It is designed for metrics like total revenue or conversions, with optional support for iROAS analysis.
*   **Strong Pre-Test Correlation**: The core assumption of TBR is that the historical relationship between the control and treatment groups will continue into the test period. The methodology works best when there is a strong, stable correlation between the two groups' performance over time. The built-in diagnostics will check for this.
