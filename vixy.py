import pandas as pd
import requests
import io
import glob
import os
from tqdm import tqdm
from tabulate import tabulate, tabulate_formats

import vix_futures_exp_dates as exp

headers = {'User-Agent': 'XYZ/3.0'}
exp = exp.run_over_time_frame(2013)

def pull():
    print('Downloading VIX Futures data...')
    reg_url = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_"
    archive_url = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/archive/VX/VX_"

    for date in tqdm(exp):
        response = requests.get(f'{reg_url}{date}.csv')
        file_object = io.StringIO(response.content.decode('utf-8'))
        df = pd.read_csv(file_object)
        if len(df) > 1:
            df.to_parquet(f'data/VIX_parquet/{date}.parquet')
    print('Done downloading VIX Futures data.')

def vix():
    pull()

    path = 'C:/Users/benib/PycharmProjects/VIX/data/VIX_parquet'
    p_files = glob.glob(os.path.join(path, "*.parquet"))

    files = []
    for i in p_files:
        files.append(pd.read_parquet(i))
    df = pd.concat(files)
    # df = df.loc[:, ~df.columns.str.contains('^Unnamed')][['Trade Date','Futures','Close']]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    d = dict()
    dex = list(df['Futures'].unique())
    for i in range(0, len(exp)):
        try: d[dex[i]] = exp[i]
        except: None

    df = df.replace({'Futures':d})
    df['VIX DTM'] = (pd.to_datetime(df['Futures']) - pd.to_datetime(df['Trade Date'])).dt.days

    df.rename(columns={'Futures':'Expiry_Date', 'VIX':'Close_Px', 'VIX DTM':'DTM'}, inplace=True)
    df = df.set_index('Trade Date')
    df = df.sort_values(by=['Trade Date', 'DTM'])

    # df = df.rename(columns={'Close':'VIX'})
    # df = df[['Trade Date', 'VIX', 'VIX DTM']]

    return df

print(tabulate(vix().tail(15),headers='keys',tablefmt=tabulate_formats[0]))
