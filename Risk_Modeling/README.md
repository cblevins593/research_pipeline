# Research On Various Models For Forecasting Portfolio Volatility

## Overview

### Research Question
Can a hybrid equity risk model combining fundamental/statistical factor covariance estimation, Ledoit-Wolf shrinkage, and exponentially weighted moving averages (EWMA) improve out-of-sample portfolio risk forecasting accuracy compared to traditional standalone approaches?

### Hypothesis
A hybrid risk model that combines fundamental or statistical factor structure, Ledoit-Wolf shrinkage, and EWMA weighting will produce more accurate and stable out-of-sample portfolio risk forecasts than models relying on any single methodology.

### Data 
Point-in-Time monthly panel tracking the S&P 500 universe retrieved from Bloomberg via BQL.


## Methods

### *Fundamental Factor Model*:<br> 
Predefined factors such as value, momentum, quality, size, and industry are used to explain co-movement between different tickers.<br>
The fundamental factor covariance matrix is defined as (F) and the portfolio fundamental factor exposures vector is defined as (B).<br> 
Residual covariance is reduced to a diagonal matrix (D) since the factor structure is assumed to account for all ticker covariance.<br>
Portfolio variance is calculated as $\Sigma = BFB^T + D$

| Strengths | Limitations |
|:---|:---|
| Reduces required covariance estimations | Potential for omitted factors |
| Economically interpretable factors | Results depend on factor construction |
| Stable, predefined factor definitions | May contain multicollinearity between factors |
| Intuitive portfolio risk attribution and exposure constraints |  |

### *PCA Factor Model*:<br>
Statistical factor model that extracts latent risk factors from historical returns to explain co-movement between different tickers.<br>
The PCA factor covariance matrix is defined as ($F_{PCA}$) and the portfolio PCA factor exposures vector is defined as ($B_{PCA}$).<br> 
Residual covariance is reduced to a diagonal matrix (D) since the factor structure is assumed to account for all ticker covariance.<br>
Portfolio variance is calculated as $\Sigma = B_{PCA}F_{PCA}B_{PCA}^T + D$

| Strengths | Limitations |
|:---|:---|
| Reduces required covariance estimations | Factors lack economic interpretations |
| Captures sources of risk not specified by the modeler | The number of factors requires parameter tuning |
| Factors have no multicollinearity  | Factor definitions may be unstable |
| Factors are selected by magnitude of historical variance |  |

### *Ledoit-Wolf Covariance Shrinkage*:
This method reduces covariance matrix estimation noise by blending the sample covariance matrix (F) with a structured target (F) e.g., the diagonal covariance matrix.<br>
The shrinkage intensity $\delta$ determines how much the sample covariance is shrunk towards the target.<br>
The shrunk covariance matrix is calculated as $\hat{\Sigma}_{LW} = (1-\delta)S + \delta F$

| Strengths | Limitations |
|:---|:---|
| Reduces noise in covariance estimations | Can distort meaningful covariance structure |
| Systematically selects optimal shrinkage intensity |  |

### *Exponential Weighting*:
Assigns exponentially greater weights to more recent observations to adapt to changing market conditions.<br>
$\lambda$ is the decay factor which is calculated as $\lambda=2^{-1/h}$

$w_i$ is the normalized exponential weight for the $i_{th}$ observation defined as:

$$
w_i=\frac{\lambda^{T-i}}{\sum_{j=1}^{T}\lambda^{T-j}}
$$

The weighted equivalent of the standard sample covariance Bessel correction $\frac{T}{(T-1)}$
is a finite sample correction factor (S), defined as 


$$
S =\frac{1}{1-\sum_{i=1}^{T}w_i^2}
$$

$$
\Sigma_t =S
\sum_{i=1}^{T}w_i
(x_i-\bar{x}_w)(x_i-\bar{x}_w)^T
$$



$$
\bar{x}_w=\sum_{i=1}^{T}w_ix_i
$$

| Strengths | Limitations |
|:---|:---|
|  | Half-life requires parameter tuning |
|  |  |
## Experimental Design
Combinations of the three methods discussed above will be tested on randomized long only portfolios. Performance will be analyzed to determine if one or more combination provides superior out of sample variance forecasts.


