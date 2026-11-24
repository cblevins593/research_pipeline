# Research On Various Models For Forecasting Portfolio Volatility
## Overview
This project compares alternative methodologies for forecasting equity portfolio variance in a monthly panel tracking the S&P 500 universe. 

The most basic variance estimate is derived from a covariance matrix formed on historical returns. This method is inherently noisy due to the shear volume of relationships to be estimated and the dynamic nature of financial markets.

A 500x500 covariance matrix needs to estimate 500*(500+1)/2 or 125,250 relationships. In a monthly panel with a 5 year estimation period, the covariances are estimated on only 60 observations. In such a small sample, certain assets may appear coorelated by pure chance.

The second limitation of the basic covariance matrix is that it treats all observations with equal weight. This may be problematic due to the dynamic nature of the equity market. Stocks that exhibited co-movement 5 years ago may no longer show the same relationship.

The methods discussed below attempt to provide more accurate variance estimates by simplifying the covariance matrix and taking into account the dynamic nature of the market to provide more stable out of sample predictions.

### Research Question
Do portfolio variance estimation methods outperform a basic covariance matrix in forecasting out of sample variance of a diversified portfolio.

### Methods
<u>**Factor Models**</u>

*Fundamental Factor Model*:<br> 
Uses predefined factors such as value, momentum, quality, size, and industry to explain co-movement between different tickers.<br>
| Pros | Cons |
|:---|:---|
| Economically interpretable factors | Potential omitted factors |
| Stable, predefined factor definitions | Results depend on how factors are constructed |
| Intuitive portfolio risk attribution | Multicollinearity between characteristics |
| Supports factor exposure constraints |  |

*PCA Factor Model*:<br>
Statistical factor model that extracts latent risk factors from historical returns to explain co-movement between different tickers
| Pros | Cons |
|:---|:---|
| Captures sources of risk not specified by the modeler | Factors lack economic interpretations |
| Factors are mathematically orthogonal | Factors are sensitive to the estimation window |
| Factors are selected by magnitude of historical variance | The modeler must determine significant components |

<u>**Shrinkage**</u>

*Ledoit-Wolf Covariance Shrinkage* -

<u>**Dynamic Adjustment**</u>

*Exponential Weighting* -

### Experiment
Combinations of the three methods discussed above will be tested on randomized long only portfolios. Performance will be analyzed to determine if one or more combination provides superior out of sample variance forecasts.


