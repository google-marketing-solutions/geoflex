# Geo-Based Regression (GBR)

The **Geo-Based Regression (GBR)** methodology is a powerful and straightforward approach to analyzing geo-experiments. At its heart, GBR uses a linear model to compare the performance of geographic areas (geos) before and during the experiment, allowing you to estimate the incremental impact of your advertising spend. This method is particularly effective when you are working with a large number of geos.

For a more detailed explanation of the methodology, see the original paper: [Measuring Ad Effectiveness Using Geo Experiments (Vaver and Koehler, 2011)](https://research.google/pubs/measuring-ad-effectiveness-using-geo-experiments/).

---
## How GBR Works

The core idea behind GBR is to model the relationship between a geo's performance in the test period and its performance in the pre-test period, while accounting for the experimental change.

### Geo Assignment

In the design phase, GeoFleX's GBR implementation assigns geos to control and treatment groups **randomly**. This is a crucial aspect of the methodology, as randomization helps to ensure that any underlying differences between the geos are, on average, balanced across the groups, which guards against potential hidden biases.

### Analysis with a Linear Model

After the experiment, GBR analyzes the results using a linear regression model:

$y_{i,1}=\beta_{0}+\beta_{1}y_{i,0}+\beta_{2}\delta_{i}+\epsilon_{i}$

Where:
* $y_{i,1}$ is the total value of the response metric (e.g., revenue) for geo *i* during the **test period**.
* $y_{i,0}$ is the total value of the response metric for geo *i* during the **pre-test period**.
* $\delta_{i}$ is the **ad spend differential** for geo *i* (the change in ad spend due to the experiment).
* $\beta_{2}$ is the coefficient of primary interest, as it represents the **Return on Ad Spend (ROAS)** for the response metric.

A key challenge in geo-experiments is that different geos have different sizes, which leads to **heteroscedasticity** (the variance of the error term is not constant). GeoFleX's GBR implementation can handle this in two ways:

1. **Weighted Least Squares (WLS)**: This is the default approach. It assumes that the variance of the response is proportional to its pre-test level. To counteract this, each geo is weighted by the inverse of its pre-test response value ($1/y_{i,0}$). If this assumption holds true, WLS can provide more statistical power and more precise estimates.

2. **Robust Ordinary Least Squares (Robust OLS)**: This method does not assume a specific structure for the heteroscedasticity. Instead, it uses standard OLS and then adjusts the standard errors of the coefficients to be robust to the presence of heteroscedasticity. While this might result in slightly less power if the WLS assumptions are perfectly met, it is a more reliable and robust method when those assumptions are violated.

In GeoFleX, you can specify which model to use via the `methodology_parameters`. However, if the conditions for WLS are not met (e.g., the data contains zero or negative values), the implementation will automatically fall back to using robust OLS to ensure the analysis can be completed successfully.

### Adapting GBR for A/B Tests (No Spend Change)

The GBR framework is also perfectly suited for standard A/B tests where there is no change in budget (e.g., testing new ad creatives). In this scenario, the concept of an ad spend differential ($\delta_{i}$) is not applicable.

GeoFleX cleverly adapts the model for this case. Instead of the spend differential, the $\delta_{i}$ term is replaced by a binary indicator variable:
* $\delta_{i} = 1$ if the geo is in the treatment group
* $\delta_{i} = 0$ if the geo is in the control group

With this change, the interpretation of the $\beta_{2}$ coefficient also changes. It no longer represents ROAS. Instead, $\beta_{2}$ directly measures the **average incremental effect** (the "lift") of the treatment on the response metric. This makes GBR a highly versatile methodology, capable of measuring both the efficiency of spend changes and the direct impact of non-spend-related changes within the same unified framework.

---
## Restrictions and Best Practices

While GBR is a robust methodology, there are a few key considerations to keep in mind:

* **Number of Geos**: GBR performs best when you have a large number of geographic areas to work with. The GeoFleX implementation requires at least 4 geos per cell on average to ensure reliable results.
* **Random Assignment is Required**: The GBR methodology in GeoFleX requires that geos be assigned randomly. It **cannot** handle experiments where specific geos are forced into the control or treatment groups. If your experiment has such constraints, you will need to choose a different methodology.
* **Data Requirements for WLS**: The default Weighted Least Squares (`wls`) model type assumes that the metric and cost values are always positive. If your data contains zeros or negative values for these columns, the GBR implementation in GeoFleX will automatically fall back to using `robust_ols` to ensure the analysis can still be run.

By understanding these principles and constraints, you can effectively leverage the Geo-Based Regression methodology in GeoFleX to gain valuable insights into the effectiveness of your advertising campaigns.