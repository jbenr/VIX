# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from tabulate import tabulate,tabulate_formats

# Press the green button in the gutter to run the script.
import vixy
# import yfin
# import berg

if __name__ == '__main__':

    # pulling data
    vix = vixy.vix()
    vix.index = pd.to_datetime(vix.index).date
    #
    # stonk = yfin.yonks()
    # stonk.index = pd.to_datetime(stonk.index).date
    #
    # print(tabulate(vix.tail(5),headers='keys',tablefmt=tabulate_formats[1]))
    # print(tabulate(stonk.tail(5),headers='keys',tablefmt=tabulate_formats[1]))
    #
    # vix.sort_index(), stonk.sort_index()
    # df = pd.merge(vix, stonk, left_index=True, right_index=True)
    # print(tabulate(df.tail(5),headers='keys',tablefmt=tabulate_formats[1]))
    #
    # bberg = berg.bberg_hist(min(df.index))
    # bberg.index = pd.to_datetime(bberg.index).date
    # df = pd.merge(df, bberg, left_index=True, right_index=True)

    # df.to_pickle('data/tester.pickle')
    # df = pd.read_pickle('data/tester.pickle')
    # df = df.drop_duplicates(subset='BTC-USD_chg', keep='first')
    #
    # print(tabulate(df.tail(9),headers='keys',tablefmt=tabulate_formats[1]))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# 20 day avg volumes