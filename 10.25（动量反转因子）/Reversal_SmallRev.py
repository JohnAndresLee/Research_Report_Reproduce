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
class Reversal_SmallRev(StockFactor):
    '''
    Reversal_SmallRev(小单反转因子)
    计算方法：回溯所有股票过去 20 个交易日的数据，
             按照小单交易占比的高低，计算小单交易占比最高的4个交易日涨幅均值和小单交易占比最低的4个交易日涨幅均值
             使用小单交易占比最低四日涨幅均值减去小单交易占比最高四日涨幅均值作为因子
    参考研报：东吴证券《20200818-“求索动量因子”系列研究（二）：交易者结构对动量因子的改进》
    '''
    def __init__(self, name, category, description,n=20):
        self.name = name
        self.category = category
        self.description = description
        self.n = n
        
    def compute(self,start_date,end_date):
        #获取前一个月的个股股价数据
        pre_date = TradeDate().get_previous_trade_dt(start_date,30)
        stock_data = EquityData().get_stocks_price(pre_date,end_date,['n_pctChange','n_volume'])
        stock_data.loc[stock_data['n_volume']==0,'n_pctChange'] = np.nan
        stock_data = stock_data.pivot(index='t_tradingDate',columns='c_code',values='n_pctChange')
        
        #获取前一个月的交易者结构数据
        moneyflow_data = EquityData().get_stocks_moneyflow(pre_date,end_date,['n_buyVolumeExlargeOrder','n_sellVolumeExlargeOrder','n_buyVolumeLargeOrder','n_sellVolumeLargeOrder','n_buyVolumeMedOrder','n_sellVolumeMedOrder','n_buyVolumeSmallOrder','n_sellVolumeSmallOrder'])
        moneyflow_data['smallorder_rate'] = moneyflow_data[['n_buyVolumeSmallOrder','n_sellVolumeSmallOrder']].sum(axis=1)/moneyflow_data[['n_buyVolumeExlargeOrder','n_sellVolumeExlargeOrder','n_buyVolumeLargeOrder','n_sellVolumeLargeOrder','n_buyVolumeMedOrder','n_sellVolumeMedOrder','n_buyVolumeSmallOrder','n_sellVolumeSmallOrder']].sum(axis=1)
        smallorder_rate = moneyflow_data.pivot(index='t_tradingDate',columns='c_code',values='smallorder_rate')
        
        #逐日计算因子值
        factor_date = TradeDate().get_trading_days(start_date,end_date)
        factor = []
        for i in factor_date:
            temp1 = stock_data[stock_data.index<=i].tail(self.n)
            
            temp2 = smallorder_rate[smallorder_rate.index<=i].tail(self.n)
            
            smallorder_mom1 = temp1[temp2.apply(lambda x:x<x.quantile(0.2))].mean()
            smallorder_mom2 = temp1[temp2.apply(lambda x:x>x.quantile(0.8))].mean()
            
            # 横截面标准化
            smallorder_mom1 = (smallorder_mom1 - smallorder_mom1.mean()) / smallorder_mom1.std()
            smallorder_mom2 = (smallorder_mom2 - smallorder_mom2.mean()) / smallorder_mom2.std()
            
            smallorder_mom = smallorder_mom1-smallorder_mom2
            
            factor.append(smallorder_mom.rename(i))
            print(i)
            
        result = pd.concat(factor,axis=1).T
        result = result.stack().rename("n_value").reset_index()
        result.columns = ['t_date','c_asset','n_value']
        result = result.sort_values(by=['t_date','c_asset'])
        return result

def get_factor(param=None):
    factor = Reversal_SmallRev(name='Reversal_SmallRev',
                            description='小单反转因子',
                            category=FactorCategory.Reversal)
    return factor

#%%测试
if __name__ == '__main__':
    start_date='20210907'
    end_date='20210910'
    
    factor = get_factor()
    a = factor.compute(start_date,end_date)
    #a.to_csv('Reversal_SmallRev.csv')
