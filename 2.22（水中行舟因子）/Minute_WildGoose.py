# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:19:46 2023

@author: Severus
"""


import time

import pandas as pd

from hs_sdk.datasource.hsbasicdata import EquityData, TradeDate
from hs_sdk.factor.base_factor import StockFactor
from hs_sdk.utils.define import FactorCategory


# %%
class Minute_WildGoose(StockFactor):
    """
    孤雁出群因子
    1.分钟市场分化度：每一分钟所有股票的分钟收益率的标准差
    2.不分化时刻：计算分钟市场分化度的均值，当日分钟市场分化度 < 均值的时刻为当日的不分化时刻
    3.取所有股票在当日不分化时刻的成交额序列，然后分别计算这些分钟里，每支股票与其他股票的分钟成交额序列的pearson相关系数（大部分个股在大部分时间并不活跃，因此使用pearson相关系数来提高交易活跃的那些分钟在计算中的权重
    4.日孤雁出群因子：对t日的相关系数矩阵，将每个个股和其他股票的相关系数取绝对值后求均值。因子越小，代表在市场越不分化时个股走出了独立趋势。
    5.孤雁出群因子：取所有股票过去20个交易日的日孤雁出群因子，计算其均值和标准差，并等权合成为孤雁出群因子
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
        :param del_minute: ignore open&close time period, default 4min
        :return: ['t_date','c_asset','n_value'] DataFrame
        """
        print(date)
        
        data = data_group.get_group(date)

        data_price = data.pivot('t_tradingTime', 'c_code', 'n_close')
        data_amount = data.pivot('t_tradingTime', 'c_code', 'n_amount')
        ret_minute = data_price.pct_change().dropna(how = 'all')
        # 确定不分化时刻
        ret_minute_std = ret_minute.apply(lambda x:x.std(), axis = 1)
        std_mean = ret_minute_std.mean()
        dif_little = pd.DataFrame(ret_minute_std < std_mean).rename(columns = {0:'dif_little'}).reset_index()
        data_amount = data_amount.reset_index().merge(dif_little)
        dif_little_amount = data_amount[data_amount['dif_little'] == True]
        amount_corr = dif_little_amount.drop('dif_little', axis = 1).set_index('t_tradingTime').corr('pearson')
        corr_abs_mean = amount_corr.abs().mean()
        
        corr = pd.DataFrame(corr_abs_mean).reset_index().rename(columns = {'index':'c_asset', 0:'n_value'})
        corr['t_date'] = date

        print(date, ' computed!')
        return corr

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
    def look_back(factor: pd.DataFrame, n, m=1):
        """
        以向前n天的平均值和方差等权 平滑当天的值
        :param factor: ['t_date','c_asset','n_value']
        :param n:
        :param m: min_periods，default 1
        :return: ['t_date','c_asset','n_value']
        """
        factor_pivot = factor.pivot(index='t_date', columns='c_asset', values='n_value').sort_index(ascending=True)
        factor_rolling = factor_pivot.rolling(window=n, min_periods=m).agg(lambda x: (x.mean() + x.std()) / 2)

        factor_melt = factor_rolling.reset_index().melt(id_vars='t_date', ignore_index=True).dropna(axis=0, how='any')
        return factor_melt.rename(columns={'value': 'n_value'})

    def compute(self, start_date, end_date):
        """
        实际提取的数据前推 self.n -1 天
        :param start_date:
        :param end_date:
        :return:
        """
        self.local_time = time.time()
        beg_date = TradeDate.get_previous_trade_dt(start_date, self.n - 1)
        trade_list = TradeDate.get_trading_days(beg_date, end_date)
        if not trade_list:
            return None
        
        extract_interval = 10
        beg_index = 0  # changing
        finish_index = len(trade_list) - 1  # fixed

        df = pd.DataFrame(columns=['t_date', 'c_asset', 'n_value'])

        while beg_index <= finish_index:
            end_index = beg_index + extract_interval if beg_index + extract_interval < finish_index else finish_index
            loop_df = self.compute_factor(trade_list[beg_index:end_index + 1])  # exclude end_index + 1
            df = pd.concat([df, loop_df], ignore_index=True)
            beg_index = end_index + 1

        modified_df = self.look_back(df, self.n, self.n)
        return modified_df.sort_values(by=['t_date', 'c_asset'])


# %%
def get_factor(param=None):
    factor = Minute_WildGoose(
        name='Minute_WildGoose',
        description='孤雁出群因子',
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
