import pandas as pd
import numpy as np
from .Risk_Utils import fundamental_factor_exposures, factor_returns, factor_covariance, residual_returns, residual_covariance, portfolio_factor_exposure, expected_factor_variance,expected_specific_variance



class risk():
    def __init__(self,
                 fundamental_factor_model:bool = False,
                 pca_factor_model:bool = False,
                 ledoit_wolf:bool = False, 
                 ewma_halflife: float|int=float('inf'),
                 
                ):
        # Initialize the factor model parameters
        self.fundamental_factor_model=fundamental_factor_model
        self.pca_factor_model=pca_factor_model
        if fundamental_factor_model and pca_factor_model:
            raise ValueError("Cannot use both fundamental factor model and PCA factor model at the same time.")
        if fundamental_factor_model or pca_factor_model:
            self.factor_model=True
            self.identity_covariance=True
            self.factor_ledoit_wolf=ledoit_wolf
        else:
            self.identity_covariance=False
            self.factor_model=False
            self.factor_ledoit_wolf=False

        # Initialize the Shrinkage method parameters
        if self.factor_model:
            self.residual_ledoit_wolf=False
        else:
            self.residual_ledoit_wolf=ledoit_wolf
        # Initialize the dyanamic covariance estimation parameters
        self.ewma_halflife=ewma_halflife

        # Initialize the risk model attributes
        self.factor_covariance=None
        self.residual_covariance=None
        self.risk_factors=None
        self.industries=None
        
    def run_model(self,historical_data:pd.DataFrame) -> None: 

        if self.factor_model:
            if self.fundamental_factor_model:
                factor_df,factors=fundamental_factor_exposures(historical_data)
            elif self.pca_factor_model:
                raise NotImplementedError("PCA factor model is not implemented yet.")
            else:
                raise ValueError("Invalid factor risk model type. Choose either fundamental_factor_model or pca_factor_model.")
            
            factor_df,factor_returns_map=factor_returns(factor_df,factors)
            factor_cov=factor_covariance(factor_df, factor_returns_map, ledoit_wolf=self.factor_ledoit_wolf,halflife=self.ewma_halflife)
        else: 
            factor_returns_map=None
            factor_cov=None
            factor_df=historical_data
            
        residual_df,residual_col_name=residual_returns(factor_df,factor_returns_map)
        residual_cov=residual_covariance(residual_df, residual_col_name, ledoit_wolf=self.residual_ledoit_wolf, identity_only=self.identity_covariance,halflife=self.ewma_halflife)
        self.factor_covariance=factor_cov
        self.residual_covariance=residual_cov


    
    def expected_portfolio_variance(self,current_data: pd.DataFrame, portfolio: pd.DataFrame) -> tuple[float,float]:
       
       
        if self.factor_model:
            if self.factor_covariance is None or self.residual_covariance is None:
                raise ValueError("Model has no covariance matrices. Please run the model on historical data first.")
            current_factor_df,factors=fundamental_factor_exposures(current_data)
            portfolio_exposures=portfolio_factor_exposure(current_factor_df,factors,portfolio)
            systematic_variance=expected_factor_variance(portfolio_exposures,factors,self.factor_covariance)
        else:
            systematic_variance=float(0)

        if self.residual_covariance is None:
                    raise ValueError("Model has no residualcovariance matrix. Please run the model on historical data first.")
        
        specific_variance=expected_specific_variance(portfolio,self.residual_covariance)

        return systematic_variance,specific_variance

    


    def realized_systematic_return(self,current_data: pd.DataFrame, portfolio: pd.DataFrame) -> float:
        if self.fundamental_factor_model:
            current_factor_df,factors=fundamental_factor_exposures(current_data)
            current_factor_df,factor_returns_map=factor_returns(current_factor_df,factors)
            portfolio_exposures=portfolio_factor_exposure(current_factor_df,factors,portfolio)
            current_factor_df=current_factor_df.iloc[0]
            systematic_return=0
            for factor in factor_returns_map:
                systematic_return+=portfolio_exposures[factor]*current_factor_df[factor_returns_map[factor]]
        else:
            systematic_return=float(0)
        return float(systematic_return)
