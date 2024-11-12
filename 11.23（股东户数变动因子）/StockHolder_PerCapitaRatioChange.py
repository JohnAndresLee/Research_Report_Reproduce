# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:50:09 2020

@author: LENOVO
"""
import pandas as pd
import numpy as np

from hs_sdk.factor.base_factor import StockFactor
from hs_sdk.utils.define import FactorCategory
from hs_sdk.datasource.hsbasicdata import EquityData,TradeDate
from hs_sdk.datasource.hs_factor_data import get_factor_data_object

def date_format(date):
    """
    将格式为yyyymmdd的字符串转变为yyyy-mm-dd的格式
    """
    strlist = list(date)
    strlist.insert(4,'-')
    strlist.insert(7,'-')
    date = ''.join(strlist)
    return date

# In[]
    
class StockHolder_PerCapitaRatioChange(StockFactor):
    """
    StockHolder_PerCapitaRatioChange(人均持股占比变动因子)
    计算方法：提取股东户数日频因子（早于当前日期发布的、且发布日期距今最近的、但截止日期距今不超过1年的公告中公布的股东户数）
             人均持股 = 当日自由流通股本 / 股东户数
             对个股人均持股间隔3个月、滚动周期为8期进行时序标准化，得到结果
    """
    
    def __init__(self, name, category, description):
        '''
        name : 因子名
        category : 因子类别
        description : 因子描述
        
        '''
        super().__init__(name, category, description)
        
    def compute(self, start_date, end_date):
        """
        计算因子值

        执行计算（如果需要）并返回所要更新的数据

        Returns
        -------
        pd.DataFrame
        
        """
        start_earlier_date = TradeDate.get_previous_trade_dt(start_date, 600)
        holdernum_day = get_factor_data_object().get_factor_data('StockHolder_HolderNum', start_earlier_date, end_date)
        freeshares = EquityData.get_stocks_valuation(start_earlier_date, end_date, ['n_freeSharesToday'])
        freeshares = freeshares.rename(columns = {'t_tradingDate':'t_date', 'c_code':'c_asset'})
        holdernum_day = holdernum_day.merge(freeshares)
        holdernum_day['PerCapita'] = holdernum_day['n_freeSharesToday'] / holdernum_day['n_value']
        holdernum_day = holdernum_day[['t_date', 'c_asset', 'PerCapita']].dropna()
        holdernum_day_pivot = holdernum_day.pivot('t_date', 'c_asset', 'PerCapita')
        
        factor_date = TradeDate.get_trading_days(start_date,end_date)
        factors = []
        for i in factor_date:
            df = holdernum_day_pivot[(holdernum_day_pivot.index <= i)]
            df = df.iloc[::-21 * 3].head(8).dropna(axis = 1, thresh = 6)
            df_temp = ((df.head(1) - df.mean()) / df.std()).dropna(axis = 1)
            df_temp = df_temp.unstack().reset_index()
            factors.append(df_temp)
            print(i)
        factors = pd.concat(factors).rename(columns = {0:'n_value'})
        return factors
    
def get_factor(param = None):
    factor = StockHolder_PerCapitaRatioChange(
            name='StockHolder_PerCapitaRatioChange',
            category=FactorCategory.StockHolder,
            description='人均持股占比变动',
    )
    return factor
    
# In[]
if __name__ == '__main__':
    start_date='20150101'
    end_date='20200101'
    
    factor = StockHolder_PerCapitaRatioChange(
            name='StockHolder_PerCapitaRatioChange',
            category=FactorCategory.StockHolder,
            description='人均持股占比变动',
    )
    factors = factor.compute(start_date,end_date)
    