from models.risk import risk_model
from models.short_interest_alpha import alpha_model
from processing.clean_data import create_risk_factors,calculate_risk_factor_returns
import pandas as pd
import numpy as np

file_name="data/SP500 Monthly Panel.xlsx"
df=pd.read_excel(file_name)
all_dates=df["Date"].unique()
print(df.columns)
clean_df,risk_factors=create_risk_factors(df)
model_df=calculate_risk_factor_returns(clean_df,risk_factors,all_dates)

print(model_df.head())

lookback=60
analytics=risk_analytics(risk_factors)
risk=risk_model(risk_factors,ret_suffix)
alpha=alpha_model()
for i, date in enumerate(all_dates):
    #print(f"\r{date}", end="", flush=True)
    if i<lookback+1:
        continue


all_dates=df["Date"].unique()