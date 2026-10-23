import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def create_risk_factors(df):
    factor_chars=["ROIC","market cap","Market-to-book","12-1 Month Momentum"]
    alpha_chars=["Short Interest % Float"]
    rel_df=df[["Date","Ticker","price","PCT_Return"]+factor_chars+alpha_chars]
    zscore=lambda x: (x - x.mean()) / x.std()
    zscore_feats=df.groupby("Date")[factor_chars+alpha_chars].transform(zscore)
    rel_df[factor_chars+alpha_chars]=zscore_feats


    risk_factors=["Value","Momentum","Quality","Size"]
    rel_df["Value"]=-rel_df["Market-to-book"]
    rel_df["Quality"]=rel_df["ROIC"]
    rel_df["Momentum"]=rel_df["12-1 Month Momentum"]
    rel_df["Size"]=-rel_df["market cap"]

    clean_df=rel_df.drop(factor_chars,axis=1)
    return clean_df,risk_factors



def calculate_risk_factor_returns(df,risk_factors,all_dates,ret_suffix=" Factor Return"):
    factor_returns={key+ret_suffix:[] for key in risk_factors}
    for i, date in enumerate(all_dates):
        
        train_data=df[df["Date"]==date].dropna()
        X=train_data[risk_factors]
        Y=train_data["PCT_Return"]
        model=LinearRegression()
        model.fit(X,Y)
        for i, factor in enumerate(risk_factors):
            factor_returns[factor+ret_suffix].append(model.coef_[i])
    factor_return_df=pd.DataFrame(factor_returns)
    factor_return_df.index=all_dates
    pit_factor_return_df=factor_return_df.shift(1)


    factor_return_df["Date"]=factor_return_df.index
    risk_model_data=df.merge(factor_return_df, on="Date", how="left")
    risk_model_data["Total Factor Return"]=sum(risk_model_data[factor] * risk_model_data[factor+ret_suffix] for factor in risk_factors)
    risk_model_data["Residual Return"]=risk_model_data["PCT_Return"]-risk_model_data["Total Factor Return"]
    return risk_model_data