import pandas as pd
import requests
import io
import glob
import os
from tqdm import tqdm
from tabulate import tabulate, tabulate_formats

import vix_futures_exp_dates as exp

exp = exp.run_over_time_frame()

headers = {'User-Agent': 'XYZ/3.0'}
reg_url = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_"

def pull():
    print('Downloading VIX Futures data...')
    for date in tqdm(exp):
        response = requests.get(f'{reg_url}{date}.csv')
        file_object = io.StringIO(response.content.decode('utf-8'))
        df = pd.read_csv(file_object)
        if len(df) > 1:
            df.to_csv(f'data/VIX/{date}.csv')
    print('Done downloading VIX Futures data.')

def vix():
    pull()

    path = 'C:/Users/benib/PycharmProjects/VIX/data/VIX'
    csv_files = glob.glob(os.path.join(path, "*.csv"))

    files = []
    for i in csv_files:
        files.append(pd.read_csv(i))
    df = pd.concat(files)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')][['Trade Date','Futures','Close']]

    d = dict()
    dex = list(df['Futures'].unique())
    print('dex',dex)
    for i in range(0, len(exp)):
        try: d[dex[i]] = exp[i]
        except: None

    df = df.replace({'Futures':d})
    # print(df.head(15))
    df['VIX DTM'] = (pd.to_datetime(df['Futures']) - pd.to_datetime(df['Trade Date'])).dt.days

    df = df.set_index('Trade Date')
    df = df.sort_values(by=['Trade Date', 'VIX DTM'])

    df = df.rename(columns={'Close':'VIX'})
    # df = df[['Trade Date', 'VIX', 'VIX DTM']]

    return df

# print(tabulate(vix(),headers='keys',tablefmt=tabulate_formats[0]))