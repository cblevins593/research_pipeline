# Research On Various Models For Forecasting Portfolio Volatility

## Overview

### Research Question
Can a hybrid equity risk model combining fundamental factor exposures, statistical covariance estimation, Ledoit-Wolf shrinkage, and exponentially weighted moving averages (EWMA) improve out-of-sample portfolio risk forecasting accuracy compared to traditional standalone approaches

### Hypothesis
A hybrid risk model that combines fundamental or statistical factor structure, Ledoit-Wolf shrinkage, and EWMA weighting will produce more accurate and stable out-of-sample portfolio risk forecasts than models relying on any single methodology.

### Data 
Point-in-Time monthly panel tracking the S&P 500 universe retrieved from Bloomberg via BQL


## Methods

### *Fundamental Factor Model*:<br> 
Predefined factors such as value, momentum, quality, size, and industry are used to explain co-movement between different tickers.<br>
Portfolio factor exposures vector is defined as (B) and fundamental factor covariance matrix as (F).<br> 
Residual covariance is reduced to an identity matrix (D) since the factor covariance is assumed to account for all ticker covariance.
Portfolio variance is calculated as follows:

$$
\Sigma = BFB^T + D
$$

| Strengths | Limitations |
|:---|:---|
| Reduces required covariance estimations | Potential for omitted factors |
| Economically interpretable factors | Results depend on factor construction |
| Stable, predefined factor definitions | May contain multicollinearity between factors |
| Intuitive portfolio risk attribution and exposure constraints |  |

### *PCA Factor Model*:<br>
Statistical factor model that extracts latent risk factors from historical returns to explain co-movement between different tickers
| Strengths | Limitations |
|:---|:---|
| Reduces required covariance estimations | Factors lack economic interpretations |
| Captures sources of risk not specified by the modeler | The number of factors requires parameter tuning |
| Factors have no multicollinearity  | Factor definitions may be unstable |
| Factors are selected by magnitude of historical variance |  |

### *Ledoit-Wolf Covariance Shrinkage*:
Shrinks sample covariance matrix (S) towards a structured target (F) e.g., the diagonal covariance matrix using the following:

$$
\hat{\Sigma}_{LW} = (1-\delta)S + \delta F
$$

| Strengths | Limitations |
|:---|:---|
| Reduces noise in covariance estimations | Can distort meaningful covariance structure |
| Systematically selects optimal shrinkage intensity |  |

### *Exponential Weighting*:
| Strengths | Limitations |
|:---|:---|
|  |  |
|  |  |
## Experimental Design
Combinations of the three methods discussed above will be tested on randomized long only portfolios. Performance will be analyzed to determine if one or more combination provides superior out of sample variance forecasts.


