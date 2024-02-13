import yfinance as yf
from datetime import datetime
from tabulate import tabulate
from functools import reduce
import pandas as pd

def yonks():
    df = yf.download(['^GSPC', '^IXIC', '^DJI', '^IRX', '^FVX', '^TNX', '^TYX', 'BTC-USD', '^VIX', 'ZN=F','ZB=F','ZF=F','ZT=F','ES=F','YM=F','NQ=F','RTY=F'], start = datetime(2013,1,1))

    df_px = df['Adj Close']
    df_vol = df['Volume']
    df_chg = df_px.pct_change(1)

    df_px.index = df_px.index.date
    df_px.columns = [i+'_px' for i in df_px.columns]

    df_chg.index = df_chg.index.date
    df_chg.columns = [i + '_chg' for i in df_chg.columns]

    df_vol.index = df_vol.index.date
    df_vol.columns = [i + '_vol' for i in df_vol.columns]

    # df_vol_20d = pd.rolling_mean(df_vol, window=5)

    # print(tabulate(df_px.tail(10),headers='keys'))

    df = pd.merge(df_px,df_chg,how='left',left_index=True,right_index=True)
    df = pd.merge(df,df_vol,how='left',left_index=True,right_index=True)

    df = df.reindex(sorted(df.columns), axis=1)
    return df

# print(tabulate(yonks().tail(10),headers='keys'))