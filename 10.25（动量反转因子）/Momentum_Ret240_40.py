# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:45:17 2021

@author: hp
"""

import pandas as pd
import numpy as np

from hs_sdk.factor.base_factor import StockFactor
from hs_sdk.utils.define import FactorCategory
from hs_sdk.datasource.hsbasicdata import EquityData,TradeDate

#%%因子计算
class Momentum_Ret240_40(StockFactor):
    '''
    Momentum_Ret240_40(小单动量因子)
    计算方法：回溯所有股票过去 240 个交易日中去掉最近的 40 个交易日的数据，
             在剩下的 200 个交易日中，先分别用每日的小单交易占比、换手率，各自划分为 2个组
             对每支股票，将其同时属于小单交易占比高组、换手率低组的交易日取出来，计算个股在这些交易日涨跌幅的平均值
    参考研报：国盛证券《“量价淘金”选股因子系列研究（二）：不同交易者结构下的动量与反转》
    '''
    def __init__(self, name, category, description):
        self.name = name
        self.category = category
        self.description = description
        
    def compute(self,start_date,end_date):
        #获取过去 240 个交易日的个股股价数据
        pre_date = TradeDate().get_previous_trade_dt(start_date, 240)
        
        stock_data = EquityData().get_stocks_yield(pre_date, end_date, ['n_pctChangeD', 'n_turnoverD'])
        stock_data.loc[stock_data['n_turnoverD']==0,'n_pctChangeD'] = np.nan
        
        #获取过去 240 个交易日的交易者结构数据
        moneyflow_data = EquityData().get_stocks_moneyflow(pre_date,end_date,['n_buyVolumeExlargeOrder','n_sellVolumeExlargeOrder','n_buyVolumeLargeOrder','n_sellVolumeLargeOrder','n_buyVolumeMedOrder','n_sellVolumeMedOrder','n_buyVolumeSmallOrder','n_sellVolumeSmallOrder'])
        moneyflow_data['smallorder_rate'] = moneyflow_data[['n_buyVolumeSmallOrder','n_sellVolumeSmallOrder']].sum(axis=1)/moneyflow_data[['n_buyVolumeExlargeOrder','n_sellVolumeExlargeOrder','n_buyVolumeLargeOrder','n_sellVolumeLargeOrder','n_buyVolumeMedOrder','n_sellVolumeMedOrder','n_buyVolumeSmallOrder','n_sellVolumeSmallOrder']].sum(axis=1)
        moneyflow_data = moneyflow_data[["c_code","t_tradingDate","smallorder_rate"]]
        
        data = pd.merge(moneyflow_data, stock_data, on = ['c_code', 't_tradingDate'])
        data.loc[data['n_pctChangeD'] > 21, 'n_pctChangeD'] = 21
        data.loc[data['n_pctChangeD'] < -21, 'n_pctChangeD'] = -21
        data = data.dropna()
        
        #逐日计算因子值
        factor_date = TradeDate().get_trading_days(start_date,end_date)
        date_list = TradeDate().get_trading_days(pre_date,end_date)
        date_list.sort()
        
        factor = []
        for i in factor_date:
            b_date = date_list[date_list.index(i) - 240]
            e_date = date_list[date_list.index(i) - 40]
            
            temp = data[(data["t_tradingDate"] < e_date)&(data["t_tradingDate"] >= b_date)]

            temp1 = temp.groupby('c_code')['smallorder_rate'].apply(lambda x: x > x.quantile(0.5))      
            temp2 = temp.groupby('c_code')['n_turnoverD'].apply(lambda x: x < x.quantile(0.5))
            temp_both = temp1 & temp2
            
            momentum = temp[temp_both].groupby('c_code')['n_pctChangeD'].mean()
            
            factor.append(momentum.rename(i))
            print(i)
            
        result = pd.concat(factor,axis=1).T
        result = result.stack().rename("n_value").reset_index()
        result.columns = ['t_date','c_asset','n_value']
        result = result.sort_values(by=['t_date','c_asset'])
        return result

def get_factor(param=None):
    factor = Momentum_Ret240_40(name='Momentum_Ret240_40',
                                description='小单动量因子',
                                category=FactorCategory.Momentum)
    return factor

#%%测试
if __name__ == '__main__':
    start_date='20210907'
    end_date='20210910'
    
    factor = get_factor()
    a = factor.compute(start_date,end_date)
    #a.to_csv('Momentum_Ret240_40.csv')
