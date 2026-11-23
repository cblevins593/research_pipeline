import pandas as pd
from pathlib import Path
def load_data():
    file_name="Data/SP500_2016-2025.xlsx"
    df=pd.read_excel(file_name)
    df['Sector'] = (df['GISC Code']//10000).astype("Int64")
    df['Industry'] = (df['GISC Code']//1000000).astype("Int64")
    return df


def remove_sparse_tickers(data,max_missing_ret=10,required_universe_size=400):
    working_df=data.pivot(index="Date",columns="Ticker",values="PCT_Return")
    threshold = len(working_df) - max_missing_ret
    working_df = working_df.dropna(axis=1, thresh=threshold)
    valid_tickers=working_df.columns.to_list()
    valid_data=data[data["Ticker"].isin(valid_tickers)]
    if len(valid_tickers)<required_universe_size:
        raise ValueError(f"Universe must contain {required_universe_size} tickers got {len(valid_tickers)}")
    return valid_data, valid_tickers