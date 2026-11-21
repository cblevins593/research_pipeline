import pandas as pd
import numpy as np
from ..Factor_Risk_Model import risk

def run_multiple_risk_models(models_dict:dict[str,risk],historical_data:pd.DataFrame)->dict[str,risk]:
    """
    Runs multiple risk models on historical data

    Parameters
    ----------
    models_dict : dict[str,risk] 
        A dictionary mapping model identifiers to their corresponding risk model instances.
    historical_data : pd.DataFrame
        A DataFrame containing the required features as columns, 
        with each row representing a stock-period observation.

    Returns
    -------
    dict[str,risk] 
        A dictionary containing the updated risk model instances after running the historical analysis.
    """
    for model in models_dict:
        models_dict[model].run_model(historical_data)
    return models_dict


class multiple_model_tracker():
    """
    Provides backtest tracking ability when using one or multiple risk models

    Parameters
    ----------
    models_dict : dict[str,risk] 
        A dictionary mapping model identifiers to their corresponding risk model instances.

    Attributes
    ----------
    variance_estimates : dict[str,dict[date,list[float]]]
        Maps each model identifier to a dictionary of observation dates and
        their corresponding portfolio variance estimates.

    systematic_variance_estimates : dict[str,dict[date,list[float]]]
        Maps each model identifier to a dictionary of observation dates and
        their corresponding systematic variance estimates.

    realized_returns : dict[date,float]
        Maps each observation date to its corresponding realized portfolio return.

    excess_returns : dict[date,float]
        Maps each observation date to its corresponding realized excess return.

    realized_systematic_returns : dict[str,dict[date,list[float]]]
        Maps each model identifier to a dictionary of observation dates and
        their corresponding realized systematic returns.

    Notes
    -------
    Variance estimates and systematic returns are stored as lists to support
    tracking multiple portfolio aggregations for each observation date.
    """
    def __init__(self,models_dict:dict[str,risk]):
        models=list(models_dict.keys())
        self.variance_estimates={key:{} for key in models}
        self.systematic_variance_estimates={key:{} for key in models}
        self.realized_returns={}
        self.excess_returns={}
        self.realized_systematic_returns={key:{} for key in models}


    # Individual Model tracking---------------------------------------------------------   
    def append_expected_variance(self,date:pd.Timestamp,model:str,variance:float,systematic_variance:float)->None:
        if date not in self.variance_estimates[model]:
            self.variance_estimates[model][date]=[variance]
        else:
            self.variance_estimates[model][date].append(variance)
            
        # Append the systematic variance estimates   
        if date not in self.systematic_variance_estimates[model]:
            self.systematic_variance_estimates[model][date]=[systematic_variance]
        else:
            self.systematic_variance_estimates[model][date].append(systematic_variance)

    def append_realized_return(self,date:pd.Timestamp,realized_return:float)->None:
        if date not in self.realized_returns:
            self.realized_returns[date]=[realized_return]
        else:
            self.realized_returns[date].append(realized_return)

    def append_excess_return(self,date:pd.Timestamp,excess_return:float)->None:
        if date not in self.excess_returns:
            self.excess_returns[date]=[excess_return]
        else:
            self.excess_returns[date].append(excess_return)

            
    def append_realized_systematic_returns(self,date:pd.Timestamp,model:str,realized_systematic_return:float)->None:
        if date not in self.realized_systematic_returns[model]:
            self.realized_systematic_returns[model][date]=[realized_systematic_return]
        else:
            self.realized_systematic_returns[model][date].append(realized_systematic_return)
            
    # ------------------------------------------------------------------------------------
    # Multiple Model Harness--------------------------------------------------------------   

    def append_multiple_variance_estimates(self,date:pd.Timestamp,models_dict:dict[str,risk],current_data:pd.DataFrame,portfolio:pd.DataFrame)->None:
        variance_dict={}
        systematic_variance_dict={}
        for model in models_dict:
            systematic_variance,residual_variance=models_dict[model].expected_portfolio_variance(current_data,portfolio)
            total_variance=systematic_variance+residual_variance
            self.append_expected_variance(date,model,total_variance,systematic_variance)
            # systematic_variance_dict[model]=systematic_variance
            # variance_dict[model]=systematic_variance+residual_variance
        # for model in variance_dict:
        #     self.append_expected_variance(date,model,variance_dict[model],systematic_variance_dict[model]) 

    def append_multiple_realized_systematic_returns(self,date:pd.Timestamp,models_dict:dict[str,risk],current_data:pd.DataFrame,portfolio:pd.DataFrame)->None:
        for model in models_dict:
            realized_systematic_return=models_dict[model].realized_systematic_return(current_data, portfolio)
            self.append_realized_systematic_returns(date,model,realized_systematic_return) 
    #---------------------------------------------------------------------------------------
    #Results Analysis-----------------------------------------------------------------------

    def standardized_returns(self,model:str)->pd.DataFrame:
        variance_expectations=pd.DataFrame(self.variance_estimates[model])
        volatility_expectations=np.sqrt(variance_expectations)
        actual_returns=pd.DataFrame(self.excess_returns)
        standardized_returns=actual_returns/volatility_expectations
        return pd.DataFrame(standardized_returns)
        
    def systematic_standardized_returns(self,model:str) -> pd.DataFrame:
        variance_expectations=pd.DataFrame(self.systematic_variance_estimates[model])
        volatility_expectations=np.sqrt(variance_expectations)
        
        actual_returns=pd.DataFrame(self.realized_systematic_returns[model])
        standardized_returns=actual_returns/volatility_expectations
        return pd.DataFrame(standardized_returns)

    def excess_kurtosis(self,data):
        n = len(data)
        if n < 4:
            return None
        mean = sum(data) / n
        # Calculate variance (second moment) and fourth moment
        variance = sum((x - mean) ** 2 for x in data) / n
        fourth_moment = sum((x - mean) ** 4 for x in data) / n
        # Calculate excess kurtosis
        if variance == 0:
            return 0
        return ((fourth_moment / (variance ** 2)) - 3)
        
    def performance_summary(self,models_dict:dict[str,risk]) -> pd.DataFrame:
        results={}

        results["Mean Z"]={model:self.standardized_returns(model).mean().mean() for model in models_dict}
        results["Standard Deviation Z"]={model:self.standardized_returns(model).mean().std() for model in models_dict} 
        results["Mean Z Squared"]={model:(self.standardized_returns(model)**2).mean().mean() for model in models_dict}
        results["Z Excess Kurtosis"]={model:self.excess_kurtosis(self.standardized_returns(model).mean()) for model in models_dict}
        # results["Mean Systematic Z"]={model:self.systematic_standardized_returns(model).mean().mean() for model in models_dict} 
        # results["Mean Systematic Z Squared"]={model:(self.systematic_standardized_returns(model)**2).mean().mean() for model in models_dict}
        return pd.DataFrame(results)


    def autocorrelation_z_squared(self,models_dict:dict[str,risk]) -> pd.DataFrame:
        results={}
        for model in models_dict:
            results[model]={f"Lag - {i+1}": (self.standardized_returns(model).mean()**2).autocorr(lag=i+1) for i in range(12)}
        return pd.DataFrame(results)


 