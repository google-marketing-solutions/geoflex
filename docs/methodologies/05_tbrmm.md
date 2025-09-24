# Time-Based Regression with Matched Markets (TBRMM)

The **Time-Based Regression with Matched Markets (TBRMM)** methodology is an advanced approach for designing and analyzing geo-experiments. It combines a sophisticated geo-assignment strategy with the robust time-series analysis of TBR.

The analysis phase of TBRMM is identical to the standard [Time-Based Regression (TBR)](/methodologies/04_tbr.md) methodology. Please refer to the TBR documentation for a detailed explanation of how the treatment effect is calculated using a time-series model.

The key innovation of TBRMM lies in its **geo assignment** process, which moves beyond simple randomization to find an optimal pairing of control and treatment groups.

Like the standard TBR implementation, the TBRMM methodology in GeoFleX is a wrapper around the powerful `matched_markets` library.

For a more detailed explanation of the matched markets geo assignment methodology, see the original paper: [A Time-based Regression Matched Markets Approach for Designing Geo Experiments (Tim Au, 2018)](https://research.google/pubs/a-time-based-regression-matched-markets-approach-for-designing-geo-experiments/).

---
## How TBRMM Works: The Matched Markets Geo Assignment

Instead of relying on random assignment, the TBRMM methodology actively searches for a partition of geos that are "well-matched." This means the resulting control and treatment groups have very similar time-series characteristics during the pre-test period, which increases the statistical power and reliability of the analysis.

This search is conducted using a **hill-climbing (or greedy) search algorithm** that iteratively explores different combinations of geos to find the best possible split.

The algorithm's goal is to optimize a score function that balances several key criteria:

1.  **High Pre-Test Correlation**: The time series of the response metric for the control and treatment groups should be as highly correlated as possible before the experiment begins.
2.  **Similar Volume**: The total volume of the response metric (e.g., total revenue) should be similar between the two groups.
3.  **Other Constraints**: The search can also incorporate other constraints, such as the number of geos in each group or the balance of a cost metric.

By running this optimization process, TBRMM aims to find a control group that serves as the best possible predictor for the treatment group's counterfactual performance. For a small number of geos (fewer than 10), the methodology can even perform an exhaustive search to find the mathematically optimal assignment.

This data-driven approach to geo assignment makes TBRMM a quasi-experimental method, as it goes beyond pure randomization to construct the best possible comparison groups from the available data.

---
## Restrictions and Best Practices

*   **Analysis is TBR**: Remember that the final analysis is still a Time-Based Regression. Therefore, all the restrictions and best practices of the [TBR methodology](/methodologies/04_tbr.md) (e.g., two-cell designs only, no inverse cost metrics) also apply to TBRMM.
*   **Computationally Intensive Design**: The search for the optimal geo assignment is computationally intensive, especially with a large number of geos. This means that the experiment design phase for TBRMM can take significantly longer than for methodologies that use simple random assignment.
*   **Quasi-Experimental Method**: Because TBRMM prioritizes finding a well-matched control group over pure randomization, it is considered a quasi-experimental (or "pseudo-experimental") method. It creates a counterfactual based on finding the best historical patterns rather than relying on the statistical assumptions of balance that come from a classic Randomized Controlled Trial (RCT).
