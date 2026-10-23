
import cvxpy as cp
import numpy as np
import pandas as pd

def portfolio_optimization(beta,alpha,risk_factors,min_position=0.01,max_position=0.05):
    factor_betas=beta
    alpha_signal=alpha.signal
    universe = alpha_signal.merge(factor_betas,on=["Ticker", "Date"],how="inner")
    universe.dropna(inplace=True)

    
    signal=universe["Signal"].to_numpy()
    exposure=universe[risk_factors].to_numpy()
    
    n = len(universe)
    w = cp.Variable(n)
    long = cp.Variable(n, boolean=True)
    short = cp.Variable(n, boolean=True)
    objective = cp.Maximize(signal @ w)
    
    constraints = [
        exposure.T @ w <= 0.1,          # factor neutral
        exposure.T @ w >= -0.1,          # factor neutral
        cp.sum(w) <= 0.05,        # dollar neutral
        cp.sum(w) >= -0.05,        # dollar neutral
        
        cp.norm1(w) <= 2,    # leverage constraint
        

        w >= min_position * long - max_position * short, #Position size constraint
        w <= max_position * long - min_position * short, #Position size constraint

        long + short <= 1, 

    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    optimal_weights = w.value
    universe["Weight"]=optimal_weights
    
    return universe 