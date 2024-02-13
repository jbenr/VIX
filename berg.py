import blpapi
from xbbg import blp
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta
from tabulate import tabulate, tabulate_formats

# print(help(blp))

dtt = datetime.utcnow()

# for i in range(1,10):
#     df = blp.bds(f'UX{i} Index',flds=['last_price','FUT_ACT_DAYS_EXP'])
#     print(
#         tabulate(df, headers='keys', tablefmt=tabulate_formats[1])
#     )

def bberg_hist(start):
    df_hist = blp.bdh(
        tickers = [f'UX{i} Index' for i in range(1,10)],
        flds = ['PX_LAST','VOLUME'],
        start_date = datetime.today().date()-relativedelta(years=10),
        end_date = datetime.today().date()
    )
    df_hist.columns = [('_'.join(col).strip()).replace(' ', '_') for col in df_hist.columns.values]
    # print(tabulate(df_hist.head(5), headers='keys', tablefmt=tabulate_formats[1]))

    for i in range(2,10):
        # print(f'{i-1}-{i}_spread')
        df_hist[f'{i-1}-{i}_spread'] = df_hist[f'UX{i}_Index_PX_LAST']-df_hist[f'UX{i-1}_Index_PX_LAST']

    # print(tabulate(df_hist.tail(5), headers='keys', tablefmt=tabulate_formats[1]))
    return df_hist

bberg_hist("2015-01-01")

# print(tick)
# df = blp.bdib(ticker=tick,time_range=(dtt-timedelta(minutes=1),dtt),dt=dtt)
# print(tabulate(df.tail(3),headers='keys',tablefmt=tabulate_formats[2]))
#
# tick = 'TYA Comdty'
# df2 = blp.bds(tick,
#               'VOL_OFF_AVG_STDDEV',
#               VOL_TERM='30D')
# print(tabulate(df2.tail(3),headers='keys',tablefmt=tabulate_formats[2]))
#
# ticks = ['TUA Cmdty','']
# df3 = blp.bdh([tick],
#               ['PX_LAST','VOL_OFF_AVG_STDDEV'],
#               start_date='20230101',
#               end_date='20230801')
# print(tabulate(df3.tail(3),headers='keys',tablefmt=tabulate_formats[2]))
#

# df = blp.bds(ticker=act,dt=dtt,time_range=dtt)
# print(blp.fut_ticker(gen_ticker=tick))
# print(tabulate(df.tail(10),headers='keys',tablefmt=tabulate_formats[1]))
# print(help(blp.bdtick))