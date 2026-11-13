import pandas as pd
import numpy as np
from .Risk_Utils import fundamental_factor_exposures, factor_returns, factor_covariance, residual_returns, residual_covariance, portfolio_factor_exposure, expected_factor_variance,expected_specific_variance



class risk():
    def __init__(self,
                 fundamental_factor_model:bool = False,
                 ledoit_wolf:bool = False, 
                 identity_covariance: bool= False,
                 factor_ewma_halflife: float|int=float('inf')
                ):
        self.fundamental_factor_model=fundamental_factor_model
        self.ledoit_wolf=ledoit_wolf
        self.identity_covariance=identity_covariance
        self.factor_ewma_halflife=factor_ewma_halflife

        
        self.factor_covariance=None
        self.residual_covariance=None
        self.risk_factors=None
        self.industries=None
        
    def run_model(self,historical_data:pd.DataFrame) -> None: 

        if self.fundamental_factor_model:
            factor_df,factors=fundamental_factor_exposures(historical_data)
            factor_df,factor_returns_map=factor_returns(factor_df,factors)
            factor_cov=factor_covariance(factor_df, factor_returns_map, halflife=self.factor_ewma_halflife)
        else: 
            factor_returns_map=None
            factor_cov=None
            factor_df=historical_data

        residual_df,residual_col_name=residual_returns(factor_df,factor_returns_map)
        residual_cov=residual_covariance(residual_df, residual_col_name, ledoit_wolf=self.ledoit_wolf, identity_only=self.identity_covariance)

        self.factor_covariance=factor_cov
        self.residual_covariance=residual_cov

    
    def expected_portfolio_variance(self,current_data: pd.DataFrame, portfolio: pd.DataFrame) -> float:
       
        if self.fundamental_factor_model:
            current_factor_df,factors=fundamental_factor_exposures(current_data)
            portfolio_exposures=portfolio_factor_exposure(current_factor_df,factors,portfolio)
            systematic_variance=expected_factor_variance(portfolio_exposures,factors,self.factor_covariance)
        else:
            systematic_variance=float(0)
        specific_variance=expected_specific_variance(portfolio,self.residual_covariance)

        return systematic_variance,specific_variance

    
    def expected_portfolio_variance(self,current_data: pd.DataFrame, portfolio: pd.DataFrame) -> float:
       
        if self.fundamental_factor_model:
            current_factor_df,factors=fundamental_factor_exposures(current_data)
            portfolio_exposures=portfolio_factor_exposure(current_factor_df,factors,portfolio)
            systematic_variance=expected_factor_variance(portfolio_exposures,factors,self.factor_covariance)
        else:
            systematic_variance=float(0)
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
