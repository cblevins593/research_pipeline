# Quantitative Research Pipeline

## Overview
This project compares multiple methods for risk modeling, alpha modeling, and portfolio optimization.<br> 
The end result is a well documented defensible quantitative research pipeline.
<br><br>
The data used is a monthly panel of point in time data of S&P 500 constituents as of the date they appear in the panel.<br>
The source for this data is Bloomberg
<br><br>


## Risk Modeling
This section explores the following methods for forecasting portfolio variance.
* Sample covariance estimation
* Fundamental factor risk model
* PCA factor risk model
* Ledoit-Wolf shrinkage
* Exponentially Weighted Moving Average (EWMA)

See [Risk_Models](Risk_Models/) for more information
## Alpha Modeling
This section compares the following methods of forecasting next month returns.
* Sample mean estimation
* OLS regression
* Ridge regression (L2 Regularization)
* Lasso regression (L1 Regularization, can remove irrelevant features)
* LightGBM (Gradient Boosting Machine)
  
See [Alpha_Models](Risk_Models/) for more information
## Portfolio Optimization
This section compares the following portfolio optimization methods given calibrated expected returns and variance estimates.
* Minimize expected variance
* Minimize expected variance subject to a minimum expected return constraint
* Maximize expected return
* Maximize expected return subject to a maximum expected variance constraint


