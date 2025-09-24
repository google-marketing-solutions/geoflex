# Synthetic Controls

The **Synthetic Controls** methodology is a quasi-experimental approach used in GeoFleX to estimate the causal impact of an intervention. Instead of relying purely on randomization, it constructs a "synthetic" version of the treatment group using a weighted combination of geos from the control group. The impact of the experiment is then measured by comparing the actual performance of the treatment group to its synthetic counterfactual.

This method is particularly powerful when it's difficult to find a single control geo or a simple random group of geos that provides a good comparison for the treatment group, but a weighted combination of them does.

For a more detailed explanation of the methodology, see the original paper: [The Economic Costs of Conflict: A Case Study of the Basque Country (Abadie and Gardeazabal, 2003)](https://business.baylor.edu/scott_cunningham/teaching/abadie-and-gardeazabal-2003.pdf).

---
## How Synthetic Controls Work

The core principle of Synthetic Controls is to find the optimal set of weights for the control group geos that allows them to best reproduce the outcomes of the treatment group before the experiment began. The weights are constrained to be non-negative and sum to 1.

### Geo Assignment

Unlike a simple random assignment, the Synthetic Controls methodology in GeoFleX actively searches for the best possible split of geos to ensure the most accurate comparison. The primary goal is to find a control group that can best predict the treatment group's performance during the pre-test period.

This is an iterative optimization process:

1.  **Iterate on Geo Splits**: The process runs for a specified number of iterations (controlled by the `num_iterations` methodology parameter). In each iteration, it creates a new, almost random assignment of geos into control and treatment groups (while respecting any pre-assigned geos).
2.  **Train a Predictive Model**: For each potential assignment, a temporary synthetic control model is trained on a portion of the historical data.
3.  **Evaluate Predictive Power**: The model's ability to predict the treatment group's performance is then evaluated on a hold-out (validation) set of historical data. This predictive power is measured using an R-squared score.
4.  **Select the Best Split**: After all iterations are complete, GeoFleX selects the geo assignment that yielded the highest predictive power (specifically, the one that maximized the minimum R-squared score across all treatment cells). This ensures that the chosen control group is a reliable predictor for the treatment group's counterfactual.

This sophisticated assignment process is designed to minimize bias and increase the statistical power of the analysis.

### Analysis

Once the experiment is complete, the analysis phase uses the chosen geo assignment to measure the treatment effect. The goal is to create a "synthetic" version of the treatment group by finding an optimal set of weights for the control group geos.

Let $Y_{Tr,t}$ be the time series of the response metric for the **treatment group**, averaged across all treatment geos at time *t*, and let $Y_{C_g, t}$ be the time series for the *g*-th geo in the **control group**. The synthetic control, $\hat{Y}_{Tr,t}$, is a weighted average of the control geos:

$\hat{Y}_{Tr,t} = \sum_{g=1}^{N_C} w_g Y_{C_g, t}$

Where:
* $N_C$ is the number of geos in the control group.
* $w_g$ are the weights assigned to each control geo, chosen such that they are all non-negative and sum to 1.

These two constraints are crucial for the robustness and interpretability of the model:

*   **Summing to 1** ensures that the synthetic control is a weighted average of the observed control geos. This prevents extrapolation and keeps the counterfactual within the convex hull of the control group's performance. In simpler terms, the model is not allowed to invent a counterfactual that is wildly different from what was actually observed in the control group.
*   **Non-negativity** prevents the model from assigning negative weights, which would be difficult to interpret (e.g., "negative geo contribution"). This constraint also acts as a form of regularization, often resulting in a sparse model where only a few, highly relevant control geos are given non-zero weights. This improves the stability of the model and makes it easier to understand which geos are most important for creating the counterfactual.

The weights are optimized to make the synthetic controls performance as close as possible to the actual treatment group's performance during the pre-test period.

Once these optimal weights are found from the pre-test data, the analysis proceeds in two main steps:

1.  **Create the Counterfactual**: The weights are applied to the control group's performance **during the test period** to create the synthetic control—the predicted counterfactual of what would have happened to the treatment group without the intervention.
2.  **Calculate the Treatment Effect**: The estimated treatment effect is the difference between the actual observed metric in the treatment group and the predicted value from the synthetic control for the test period. Statistical significance is then calculated based on this difference.

This entire process is powered by the [`pysyncon`](https://sdfordham.github.io/pysyncon/) library under the hood.

---
## Restrictions and Best Practices

*   **Computationally Intensive**: The search for the optimal geo assignment involves training and evaluating many models, which can be computationally intensive and time-consuming. The `num_iterations` parameter allows you to control the trade-off between the exhaustiveness of the search and the speed of the exploration.
*   **Supports Pre-Assignments**: Unlike the GBR methodology, Synthetic Controls in GeoFleX can accommodate experiments where specific geos are forced into control or treatment groups. The optimization process will work with the remaining available geos to find the best possible control group.
*   **Quasi-Experimental Method**: It's important to remember that Synthetic Controls is a quasi-experimental (or "pseudo-experimental") method. It creates a counterfactual based on historical patterns rather than relying on the statistical assumptions of balance that come from pure randomization in a classic Randomized Controlled Trial (RCT).