# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:31:19 2023

@author: Severus
"""


import time

import pandas as pd

from hs_sdk.datasource.hsbasicdata import EquityData, TradeDate
from hs_sdk.factor.base_factor import StockFactor
from hs_sdk.utils.define import FactorCategory


# %%
class Minute_Drift(StockFactor):
    """
    随波逐流因子
    1.合理收益率：计算股票A的第t、t-1、...、t-19共20个交易日的日内收益率（收盘/开盘-1）的均值，记为A的第t日的“合理收益率”
    2.相对开盘收益率：开盘后，每分钟的分钟收盘价/当日开盘价-1
    3.全天240分钟里，将相对开盘收益率 > 当日合理收益率的分钟记为“高位时刻”，相对开盘收益率 < 当日合理收益率的分钟记为“低位时刻”
    4.高位成交额：t日中高位时刻的成交额之和
      低位成交额：t日中低位时刻的成交额之和
      高低额差：（高位成交额 - 低位成交额） / （高位成交额 + 低位成交额）
    5.取所有股票过去20个交易日的高低额差序列，计算个股与其余所有股票的高低额差序列之间的spearman相关系数（这里不强调指标大小，主要考虑和市场走势关联程度）
    6.随波逐流因子：将上述相关系数取绝对值，再计算股票A与其余股票的相关系数的均值，记为当日的随波逐流因子。结果越大，个股这段时间成交额和市场趋势关联越强。
    """

    def __init__(self, name, category, description, n):
        """
        初始化
        :param name: 因子名称
        :param category: 分类
        :param description: 描述
        :param n: 回看天数
        """
        super().__init__(name, category, description)
        self.n = max(n, 3)

    def compute_one_day(self, data_group, date):
        """
        计算一天的因子值
        :param data_group: all ['c_code','t_tradingDate','t_tradingTime','n_volume','n_close',...] groupby('t_tradingDate')
        :param date: 'yyyy-mm-dd'
        :return: ['t_date','c_asset','n_value'] DataFrame
        """
        print(date)
        
        data = data_group.get_group(date)
        
        # 计算相对开盘收益率
        data_price = data.pivot('t_tradingTime', 'c_code', 'n_close')
        relative_open_return = data_price / data_price.iloc[0, :] - 1
        relative_open_return = relative_open_return.unstack().reset_index().rename(columns = {0:'relative_open_return'})
        minute_data = data.merge(relative_open_return).merge(retday_ave)
        
        # 标记高位低位时刻，并计算高低额差
        minute_data['high'] = minute_data['relative_open_return'] > minute_data['ret_ave']
        high_amount = minute_data[minute_data['high'] == True].groupby('c_code')['n_amount'].sum().reset_index().rename(columns = {'n_amount':'high_amount'})
        low_amount = minute_data[minute_data['high'] == False].groupby('c_code')['n_amount'].sum().reset_index().rename(columns = {'n_amount':'low_amount'})
        dif_per_amount = high_amount.merge(low_amount)
        dif_per_amount['dif'] = ((dif_per_amount['high_amount'] - dif_per_amount['low_amount']) /
                                 (dif_per_amount['high_amount'] + dif_per_amount['low_amount']))
        dif_per_amount = dif_per_amount.drop(['high_amount', 'low_amount'], axis = 1)
        dif_per_amount['t_date'] = date

        print(date, ' computed!')
        return dif_per_amount

    def compute_factor(self, d_list):
        """
        sub process program
        :param d_list: each element in 'yyyy-mm-dd' format, get directly from TradeDate.get_trading_days
        :return:
        """
        print(f'start_date: {d_list[0]}\t end_date: {d_list[-1]}\t')
        fields = ['n_close', 'n_amount']
        d_list = [k.replace('-', '') for k in d_list]
        df = pd.DataFrame(columns=['t_date', 'c_asset', 'n_value'])

        data = EquityData.get_stocks_marketminute(d_list[0], d_list[-1], fields)
        data.drop_duplicates(keep='first', inplace=True)
        print(
            f'start_date: {d_list[0]}\t end_date: {d_list[-1]}\t Minute data fetched, using {time.time() - self.local_time:.5f}s')

        date_list = data['t_tradingDate'].unique()
        data = data.groupby(['t_tradingDate'])
        for date in date_list:
            df_day = self.compute_one_day(data, date)
            df = pd.concat([df, df_day], ignore_index=True)
        print(f'start_date: {d_list[0]}\t end_date: {d_list[-1]}\t run time is {time.time() - self.local_time:.5f}s')
        return df

    @staticmethod
    def cal_corr(dif_per_amount: pd.DataFrame, n, m = 15):
        """
        计算计算个股与其他股票向前n天的高低差额的相关性
        :param factor: ['t_date','c_asset','n_value']
        :param n:
        :param m: min_periods，default 15
        :return: ['t_date','c_asset','n_value']
        """
        
        dif_pivot = dif_per_amount.pivot('t_date', 'c_code', 'dif')
        dif_corr = dif_pivot.rolling(n, m).corr().dropna(how = 'all')    # 每n个交易日计算相关系数
        abs_corr = dif_corr.abs()
        print(abs_corr)
        print(abs_corr.reset_index())
        abs_corr_mean = abs_corr.reset_index().groupby('level_0').mean().drop('level_1', axis = 1)  # 绝对值后求平均
        abs_corr_mean.columns.name = 'c_asset'
        abs_corr_mean.index.name = 't_date'
        result = abs_corr_mean.unstack().reset_index().rename(columns = {0:'n_value'})
        return result

    def compute(self, start_date, end_date):
        """
        实际提取的数据前推 self.n-1 天
        :param start_date:
        :param end_date:
        :return:
        """
        self.local_time = time.time()
        beg_date = TradeDate.get_previous_trade_dt(start_date, self.n - 1)
        trade_list = TradeDate.get_trading_days(beg_date, end_date)
        if not trade_list:
            return None

        # 计算日内合理收益率并作为全局变量在其他函数中直接调用
        global retday_ave
        fields = ['n_close', 'n_open']
        price = EquityData().get_stocks_price(beg_date, end_date, fields)
        price['return_day'] = price['n_close'] / price['n_open'] - 1
        retday_ave_pivot = price.pivot('t_tradingDate', 'c_code', 'return_day').rolling(19, 10).mean()
        retday_ave = retday_ave_pivot.unstack().reset_index().rename(columns = {0:'ret_ave'})
        startdate_1 = start_date[:4] + '-' + start_date[4:6] + '-' + start_date[6:]
        enddate_1 = end_date[:4] + '-' + end_date[4:6] + '-' + end_date[6:]
        retday_ave = retday_ave[(retday_ave['t_tradingDate'] >= startdate_1) & (retday_ave['t_tradingDate'] <= enddate_1)]
        
        extract_interval = 10
        beg_index = 0  # changing
        finish_index = len(trade_list) - 1  # fixed

        df = pd.DataFrame(columns=['t_date', 'c_asset', 'n_value'])

        while beg_index <= finish_index:
            end_index = beg_index + extract_interval if beg_index + extract_interval < finish_index else finish_index
            loop_df = self.compute_factor(trade_list[beg_index:end_index + 1])  # exclude end_index + 1
            df = pd.concat([df, loop_df], ignore_index=True)
            beg_index = end_index + 1

        modified_df = self.cal_corr(df, self.n, self.n)
        return modified_df.sort_values(by=['t_date', 'c_asset'])


# %%
def get_factor(param=None):
    factor = Minute_Drift(
        name='Minute_Drift',
        description='随波逐流因子',
        category=FactorCategory.Minute,
        n=20
    )
    return factor


# %%
if __name__ == '__main__':
    fac = get_factor()
    start_date = '20230209'
    end_date = '20230211'
    fac_value = fac.compute(start_date, end_date)
