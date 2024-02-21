# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from tabulate import tabulate,tabulate_formats

import vixy
import yfin
# import berg

if __name__ == '__main__':

    # pulling data
    # vix = vixy.vix()
    vix = pd.read_parquet('data/tester.parquet')
    # print(tabulate(vix.head(50),headers='keys'))
    # vix.to_parquet('data/tester.parquet')

    df = vix[['Trade_Date','Expiry_Date','Close','DTM','Calendar_Num']]

    pivot_df = df.pivot_table(index='Trade_Date', columns='Calendar_Num', values='Close', aggfunc='first')
    pivot_df.columns = [f"{col}_Close" for col in pivot_df.columns]

    pivot_df_ = df.pivot_table(index='Trade_Date', columns='Calendar_Num', values='DTM', aggfunc='first')
    pivot_df_.columns = [f"{col}_DTM" for col in pivot_df_.columns]

    pivot_df = pd.merge(pivot_df, pivot_df_, how='left', on='Trade_Date')

    pivot_df.reset_index(inplace=True)
    pivot_df = pivot_df[[
        'Trade_Date', '1.0_DTM', '1.0_Close', '2.0_Close', '3.0_Close',
        '4.0_Close', '5.0_Close', '6.0_Close', '7.0_Close', '8.0_Close'
    ]].set_index('Trade_Date')

    # Iterate over each column
    for i in range(1, len([i for i in pivot_df.columns if 'Close' in i])):
        # Iterate over subsequent columns
        for j in range(i + 1, len([i for i in pivot_df.columns if 'Close' in i])):
            # Create a new column for the spread between ith column and the jth column
            pivot_df[f'{i}-{j}'] = pivot_df[pivot_df.columns[i]] - pivot_df[pivot_df.columns[j]]

    # print(pivot_df.head(3))
    print(tabulate(pivot_df.tail(3),headers='keys',tablefmt=tabulate_formats[1]))


    # stonk = yfin.yonks()
    #
    # pivot_df.index = pd.to_datetime(pivot_df.index).date
    # stonk.index = pd.to_datetime(stonk.index).date
    # df = pd.merge(stonk[['^VIX_px']], pivot_df, how='right', left_index=True, right_index=True)
    #
    # print(tabulate(df.tail(30),headers='keys',tablefmt=tabulate_formats[1]))


    # bberg = berg.bberg_hist(min(df.index))
    # bberg.index = pd.to_datetime(bberg.index).date
    # df = pd.merge(df, bberg, left_index=True, right_index=True)

    # df.to_pickle('data/tester.pickle')
    # df = pd.read_pickle('data/tester.pickle')
    # df = df.drop_duplicates(subset='BTC-USD_chg', keep='first')
    #
    # print(tabulate(df.tail(9),headers='keys',tablefmt=tabulate_formats[1]))

# 20 day avg volumes