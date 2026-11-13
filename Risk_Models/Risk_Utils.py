import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf

#Fundamental factor model----------------------------------------------------------------------------
def fundamental_factor_exposures(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    z_factor_chars=["ROIC","market cap","Market-to-book","12-1 Month Momentum","Market Beta 1 Year"]
    industry_char="Sector"
    alignment_keys=["Date","Ticker"]
    target_keys=["PCT_Return"]
    
    factor_df=df[alignment_keys+target_keys+z_factor_chars]
    zscore=lambda x: (x - x.mean()) / x.std()
    zscore_feats=factor_df.groupby("Date")[z_factor_chars].transform(zscore)
    factor_df[z_factor_chars]=zscore_feats

    
    factor_df["Value"]=-factor_df["Market-to-book"]
    factor_df["Quality"]=factor_df["ROIC"]
    factor_df["Momentum"]=factor_df["12-1 Month Momentum"]
    factor_df["Size"]=-factor_df["market cap"]
    factor_df.drop(z_factor_chars,axis=1,inplace=True)

    industry_exposures=pd.get_dummies(df[industry_char], dtype=int)
    industry_codes=industry_exposures.columns.to_list()
    industry_codes_str=[str(code) for code in industry_codes]
    
    factor_df[industry_codes_str]=industry_exposures
    factors = [col for col in factor_df.columns if col not in alignment_keys+target_keys]
               
    return factor_df,factors


def factor_returns(df: pd.DataFrame, factors: list[str]) -> tuple[pd.DataFrame, dict[str,str]]:
    ret_suffix=" Factor Return"
    factor_returns={key+ret_suffix:[] for key in factors}
    factor_returns_map={key:key+ret_suffix for key in factors}

    all_dates=df["Date"].unique()
    for date in all_dates:
        train_data=df[df["Date"]==date].dropna()
        
        X=train_data[factors]
        Y=train_data["PCT_Return"]
        
        model=LinearRegression(fit_intercept=False)
        model.fit(X,Y)
        
        for i, factor in enumerate(factors):
            factor_returns[factor+ret_suffix].append(model.coef_[i])
            
    factor_df=pd.DataFrame(factor_returns)
    factor_df["Date"]=all_dates
    factor_df=df.merge(factor_df, on="Date", how="left")
    return factor_df, factor_returns_map


def factor_covariance(factor_df: pd.DataFrame, factor_returns_map: dict[str,str],halflife:int|float=float('inf')) -> pd.DataFrame:
        factors=factor_returns_map.keys()
        factor_return_cols=[factor_returns_map[factor] for factor in factors]
    
        collapsed_df = (factor_df.groupby("Date")[factor_return_cols].first())
        demeaned_returns=collapsed_df-collapsed_df.mean().to_numpy()

        
        decay=.5**(1/halflife)
        
        periods_past=np.arange(len(demeaned_returns))[::-1]
        exponential_weights = decay ** periods_past
        covariance_matrix =  (exponential_weights * demeaned_returns.T) @ demeaned_returns / exponential_weights.sum()
        covariance_matrix.index=factors
        covariance_matrix.columns=factors
        
        return covariance_matrix
#------------------------------------------------------------------------------------------------------------------------
# Specific variance model functions------------------------------------------------------------------------------------------------
def residual_returns(df: pd.DataFrame, factor_return_map: dict[str,str] | None = None) -> tuple[pd.DataFrame,str]:
    residual_column="Residual Return" 
    if factor_return_map is None:
        df[residual_column]=df["PCT_Return"]
        return df, residual_column
    
    factors=factor_return_map.keys()
    total_factor_return=sum((df[factor] * df[factor_return_map[factor]]).fillna(0) for factor in factors) #verify this is reasonable
    df[residual_column]=df["PCT_Return"]-total_factor_return
    
    return df, residual_column

def residual_covariance(residual_returns:pd.DataFrame,residual_column: str, ledoit_wolf: bool = False,identity_only: bool = False) -> pd.DataFrame:
    returns_wide = residual_returns.pivot(index="Date",columns="Ticker",values=residual_column)
    returns_wide=returns_wide.fillna(0)
    if ledoit_wolf:
        tickers=returns_wide.columns
        lw = LedoitWolf(assume_centered=False)
        lw.fit(returns_wide.fillna(0))
        covariance_matrix = lw.covariance_
        covariance_matrix=pd.DataFrame(covariance_matrix,index=tickers,columns=tickers)
    else:
        covariance_matrix=returns_wide.cov(min_periods=10)
        
    if identity_only:
        diagonal_array=np.diag(np.diag(covariance_matrix))
        covariance_matrix=pd.DataFrame(diagonal_array, index=covariance_matrix.index, columns=covariance_matrix.columns)
    return covariance_matrix

#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Portfolio Variance Calculation Functions

def portfolio_factor_exposure(current_factor_data: pd.DataFrame,factors:list[str],portfolio:pd.DataFrame) -> pd.Series:
    portfolio_exposures={}
    combined_df=portfolio.merge(current_factor_data,on=["Date","Ticker"],how="left")
    for factor in factors:
        individual_exposures=combined_df[factor]*combined_df["Weight"]
        portfolio_exposures[factor]=individual_exposures.sum()
    return(pd.Series(portfolio_exposures))

    
def expected_factor_variance(factor_exposures:pd.Series,factors:list[str],factor_covariance:pd.DataFrame) -> float:
    
    factors=factor_exposures.index.tolist()
    factor_covariance_array=np.array(factor_covariance.loc[factors,factors])
    tot_var=factor_exposures@factor_covariance_array@factor_exposures

    return float(tot_var)
    

def expected_specific_variance(portfolio: pd.DataFrame, residual_covariance: pd.DataFrame) -> float:
    tickers=portfolio["Ticker"]
    aligned_covariance=np.array(residual_covariance.loc[tickers,tickers])
    specific_variance=portfolio["Weight"]@aligned_covariance@portfolio["Weight"]
    return float(specific_variance)




    




























