# Quantitative Research Pipeline

## Overview
This project compares multiple methods for risk modeling, alpha modeling, and portfolio optimization.<br> 
The end result is a well documented defensible quantitative research pipeline.
<br><br>
<ins> Data </ins>
* Sourced from Bloomberg
* Monthly Point-in-Time panel of S&P 500 constituents as of the date they appear in the index


## Risk Modeling
This section explores the following methods for forecasting portfolio variance.
* Sample covariance estimation
* Fundamental factor risk model
* PCA factor risk model - Marchenko-Pastur denoising
* Ledoit-Wolf shrinkage
* Exponentially Weighted Moving Average (EWMA)

See [Risk_Models](Risk_Models/) for more information.
## Alpha Modeling
This section compares the following methods of forecasting next month returns.
* Sample mean estimation
* OLS regression
* Ridge regression (L2 Regularization)
* Lasso regression (L1 Regularization, can remove irrelevant features)
* LightGBM (Gradient Boosting Machine)
  
See [Alpha_Models](Alpha_Models/) for more information.
## Portfolio Optimization
This section compares the following portfolio optimization methods given calibrated expected returns and variance estimates.
* Minimize expected variance
* Minimize expected variance subject to a minimum expected return constraint
* Maximize expected return
* Maximize expected return subject to a maximum expected variance constraint

See [Portfolio_Optimizer](Portfolio_Optimizer/) for more information.

