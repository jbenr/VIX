import yfinance as yf
from datetime import datetime
from tabulate import tabulate
from functools import reduce
import pandas as pd

def yonks():
    df = yf.download(['^GSPC', '^IXIC', '^DJI', 'IWM', '^FVX', '^TNX', '^TYX', 'BTC-USD', '^VIX',
                      "IYC", "IYK", "IYE", "IYF", "IYH", "IYJ", "IYM", "IYW", "IYZ", "IDU"], start = datetime(2013,1,1))

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

    # df = pd.merge(df_px,df_chg,how='left',left_index=True,right_index=True)
    # df = pd.merge(df,df_vol,how='left',left_index=True,right_index=True)

    df = df_px.reindex(sorted(df_px.columns), axis=1)
    df = df.rename_axis('Trade_Date')

    return df

# print(tabulate(yonks().tail(10),headers='keys'))