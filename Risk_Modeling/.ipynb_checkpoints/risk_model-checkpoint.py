import pandas as pd
import numpy as np

class risk_model:
    def __init__(self,risk_factors,suffix=" Factor Return"):
        self.risk_factors=risk_factors
        self.risk_factor_return_cols=factor_ret_cols=[x+suffix for x in risk_factors]
        self.risk_df=None
        self.factor_covariance=None
        self.residual_var=None
        
    def ingest(self,risk_data):
        self.risk_df=risk_data
        
    def calc_factor_covariance(self):
        if self.risk_df is None:
            return None

        collapsed_df = (self.risk_df.groupby("Date")[self.risk_factor_return_cols].first())
        cov_df=collapsed_df.cov()
        cov_df.index=self.risk_factors
        cov_df.columns=self.risk_factors
        self.factor_covariance=cov_df

        
    def calc_residual_var(self,min_obs=20):
        if self.risk_df is None:
            return None
        residuals=self.risk_df[["Date","Ticker","Residual Return"]]
        residual_wide = residuals.pivot(index="Date",columns="Ticker",values="Residual Return")
        residual_wide=residual_wide.dropna(axis=1,thresh=min_obs) #remove columns with less than 30 real values
        residual_var=residual_wide.var().rename("Residual Var")
        self.residual_var=residual_var
        
    def calc_raw_covariance_matrix(self):
        if self.risk_df is None:
            return None
        returns_wide = self.risk_df.pivot(index="Date",columns="Ticker",values="PCT_Return")
        return returns_wide.cov()
        


    def port_factor_exposure(self,portfolio):
        factor_exposure=pd.Series()
        for factor in self.risk_factors:
            partial_df=portfolio[factor]*portfolio["Weight"]
            factor_exposure[factor]=partial_df.sum().item()
        return factor_exposure

    def port_factor_variance(self,portfolio):
        factor_variance={}
        factor_exposures=self.port_factor_exposure(portfolio)
        for factor in self.risk_factors:
            factor_cov=self.factor_covariance[factor]
            factor_beta=factor_exposures[factor]
            risk_contribution=(factor_exposures*factor_cov*factor_beta).sum()
            factor_variance[factor]=risk_contribution
        tot_var=factor_exposures@self.factor_covariance@factor_exposures
        factor_variance["All_Factors"]=tot_var
        factor_risk=pd.Series(factor_variance)

        return factor_risk

    def port_specific_variance(self,portfolio):
        port_var=portfolio[["Ticker","Weight"]].merge(self.residual_var,on=["Ticker"],how="outer")
        port_var["Partial_Var"]=port_var["Weight"]**2*port_var["Residual Var"]
        specific_var=port_var["Partial_Var"].sum()
        
        return specific_var
        
    
            
    def run_model(self,historical_data):
        self.ingest(historical_data)
        self.calc_factor_covariance()
        self.calc_residual_var()