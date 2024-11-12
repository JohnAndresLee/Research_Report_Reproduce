# -*- coding: utf-8 -*-
"""
Created on 2022/9/23 15:44

@author: Severus
"""
import time

import pandas as pd
import numpy as np
import math
from hs_sdk.datasource.hsbasicdata import EquityData, TradeDate
from hs_sdk.factor.base_factor import StockFactor
from hs_sdk.utils.define import FactorCategory


# %%
class Volatility_LargeJumpNegative(StockFactor):
    """
    大程下行跳跃波动因子
    计算方法：
    1. 大程跳跃的阈值gamma：gamma = alpha * pow(N, -0.49) * pow(IV, 0.5)
    2. 大程下行跳跃序列：对数收益率 < -阈值 gamma
    3. 大程下行跳跃波动：大程下行跳跃序列的对数收益平方和
    参考研报：广发证券报告《基于股价跳跃模型的因子研究——高频数据因子研究系列九》
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
        :param data_group: all ['','t_tradingDate','t_tradingTime','n_volume','n_close',...] groupby('t_tradingDate')
        :param date: 'yyyy-mm-dd'
        :return: ['t_date','c_asset','n_value'] DataFrame
        """
        print(date)
        data = data_group.get_group(date)
        fields = ['n_close']
        data_p_v = data.pivot(index='t_tradingTime', columns='c_code', values=fields)
        data_price = data_p_v[fields[0]]
        
        # RJV_t
        log_price = np.log(data_price)
        log_return = (log_price - log_price.shift(5)).dropna()
        log_return = log_return.iloc[::5].reset_index().drop('t_tradingTime', axis = 'columns')
        log_return_sqr = np.square(log_return)
        RV_t = log_return_sqr.apply(lambda x: x.sum())
        RV_t = pd.DataFrame(RV_t)
        k = 3
        m = 2 / k
        miu = pow(2, m/2) * math.gamma((k + 1) / 2) * math.gamma(1/2)

        # IV_t = abs(log_return).rolling(3, 3).apply(lambda x:np.cumprod(x)[-1], raw = True).dropna().sum()  
        # 与下面等价
        abs_log_return = abs(log_return)
        abs_log_return = pow(abs_log_return, m)
        log_return_shift1 = abs_log_return.shift(1)
        log_return_shift2 = abs_log_return.shift(2)
        IV_t = (abs_log_return * log_return_shift1 * log_return_shift2).dropna().sum()

        IV_t = pd.DataFrame(IV_t * pow(miu, -2/m))
        RJV_t = RV_t - IV_t
        RJV_t[RJV_t < 0] = 0
        RJV_t = RJV_t.reset_index().rename(columns={0: 'n_value'})

        # SJ_t
        log_return_positive = log_return > 0
        RV_t_positive = (log_return_positive * log_return_sqr).sum()
        RV_t_positive = pd.DataFrame(RV_t_positive)
        log_return_negative = log_return < 0
        RV_t_negative = (log_return_negative * log_return_sqr).sum()
        RV_t_negative = pd.DataFrame(RV_t_negative)
        SJ_t = RV_t_positive - RV_t_negative
        SJ_t = SJ_t.reset_index().rename(columns={0: 'n_value'})

        # RJVP_t, RJVN_t, SRJV_t
        RJVP_t = RV_t_positive - 0.5 * IV_t
        RJVP_t[RJVP_t < 0] = 0
        RJVN_t = RV_t_negative - 0.5 * IV_t
        RJVN_t[RJVN_t < 0] = 0
        SRJV_t = RJVP_t - RJVN_t
        RJVP_t = RJVP_t.reset_index().rename(columns={0: 'n_value'})
        RJVN_t = RJVN_t.reset_index().rename(columns={0: 'n_value'})
        SRJV_t = SRJV_t.reset_index().rename(columns={0: 'n_value'})

        # gamma
        N = len(log_return)
        alpha = 4
        gamma = alpha * pow(N, -0.49) * np.sqrt(IV_t)
        
        # RLJV_t, RSJV_t
        log_big_sqr = ((abs(log_return) - np.array(gamma.T) > 0) * log_return_sqr).sum()
        com_less = np.array(RJV_t['n_value']) < np.array(log_big_sqr)
        com_more = np.array(RJV_t['n_value']) >= np.array(log_big_sqr)
        RLJV_t = com_less * np.array(RJV_t['n_value']) + com_more * np.array(log_big_sqr)
        RLJV_t = pd.DataFrame(RLJV_t)
        RLJV_t['c_code'] = RJV_t['c_code']
        RLJV_t = RLJV_t.rename(columns={0: 'n_value'})
        RSJV_t = RJV_t.copy(deep = True)
        RSJV_t['n_value'] = RSJV_t['n_value'] - RLJV_t['n_value']       # 感觉代码没啥问题, 但4628/4722都是0

        # RLJVP_t, RLJVN_t, SRLJV_t
        judge_bigup = log_return - np.array(gamma.T)
        judge_bigup = judge_bigup >= 0
        log_bigup_sqr = (judge_bigup * log_return_sqr).sum()
        com_less = np.array(RJVP_t['n_value']) < np.array(log_bigup_sqr)
        com_more = np.array(RJVP_t['n_value']) >= np.array(log_bigup_sqr)
        RLJVP_t = com_less * np.array(RJVP_t['n_value']) + com_more * np.array(log_bigup_sqr)
        RLJVP_t = pd.DataFrame(RLJVP_t)
        RLJVP_t['c_code'] = RJVP_t['c_code']
        RLJVP_t = RLJVP_t.rename(columns={0: 'n_value'})

        judge_bigdown = log_return + np.array(gamma.T)
        judge_bigdown = judge_bigdown <= 0
        log_bigdown_sqr = (judge_bigdown * log_return_sqr).sum()
        com_less = np.array(RJVN_t['n_value']) < np.array(log_bigdown_sqr)
        com_more = np.array(RJVN_t['n_value']) >= np.array(log_bigdown_sqr)
        RLJVN_t = com_less * np.array(RJVN_t['n_value']) + com_more * np.array(log_bigdown_sqr)
        RLJVN_t = pd.DataFrame(RLJVN_t)
        RLJVN_t['c_code'] = RJVN_t['c_code']
        RLJVN_t = RLJVN_t.rename(columns={0: 'n_value'})

        # Rename
        RLJVN_t.rename(columns = {'c_code':'c_asset'}, inplace = True)
        RLJVN_t['t_date'] = date
        
        print(date, ' computed!')
        return RLJVN_t

    def compute_factor(self, d_list):
        """
        sub process program
        :param d_list: each element in 'yyyy-mm-dd' format, get directly from TradeDate.get_trading_days
        :return:
        """
        print(f'start_date: {d_list[0]}\t end_date: {d_list[-1]}\t')
        fields = ['n_close']
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
        因子值
        :param start_date:
        :param end_date:
        :return:
        """
        self.local_time = time.time()
        beg_date = TradeDate.get_previous_trade_dt(start_date, self.n - 1)
        trade_list = TradeDate.get_trading_days(beg_date, end_date)
        if not trade_list:
            print("None")
            return None

        extract_interval = 10
        beg_index = 0  # changing
        finish_index = len(trade_list) - 1  # fixed

        df = pd.DataFrame(columns=['t_date', 'c_asset', 'n_value'])

        while beg_index <= finish_index:
            end_index = beg_index + extract_interval if beg_index + extract_interval < finish_index else finish_index
            loop_df = self.compute_factor(trade_list[beg_index:end_index + 1])  # exclude end_c_code + 1
            df = pd.concat([df, loop_df], ignore_index=True)
            beg_index = end_index + 1

        modified_df = self.look_back(df, self.n, self.n)
        return modified_df.sort_values(by=['t_date', 'c_asset'])


# %%
def get_factor(param=None):
    factor = Volatility_LargeJumpNegative(name='Volatility_LargeJumpNegative'
                                    , category=FactorCategory.Volatility
                                    , description='大程下行跳跃波动因子'
                                    , n = 3)
    return factor


# %%
if __name__ == '__main__':
    fac = get_factor()
    start_date = '20200921'
    end_date = '20200927'
    fac_value = fac.compute(start_date, end_date)
    fac_value = fac_value.reset_index().drop('index', axis = 'columns')
    print(fac_value)