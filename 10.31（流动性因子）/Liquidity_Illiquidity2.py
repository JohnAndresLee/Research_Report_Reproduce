# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:45:17 2021

@author: hp
"""

import pandas as pd
import numpy as np
from dateutil.parser import parse

from hs_sdk.factor.base_factor import StockFactor
from hs_sdk.utils.define import FactorCategory
from hs_sdk.datasource.hsbasicdata import EquityData,TradeDate

#%%因子计算
class Liquidity_Illiquidity2(StockFactor):
    '''
    Liquidity_Illiquidity2(改进非流动性因子)
    计算方法：回溯过去20天的日收益率和交易额
             对个股的20天日收益率绝对值与交易额的倒数序列，求协方差
    参考研报：中信建投证券《流动性因子系统解读与再增强》
    '''
    def __init__(self, name, category, description, n=20):
        self.name = name
        self.category = category
        self.description = description
        self.n = n
        
    def compute(self,start_date,end_date):
        #获取前一个月的个股股价数据
        pre_date = TradeDate().get_previous_trade_dt(start_date, self.n + 10)
        stock_data = EquityData().get_stocks_price(pre_date, end_date, ['n_pctChange','n_amount'])
        stock_data.loc[stock_data['n_amount']==0,'n_pctChange'] = np.nan
        stock_data = stock_data.dropna()
        stock_data['n_pctChange'] = abs(stock_data['n_pctChange'])
        stock_data['n_amount'] = 1 / stock_data['n_amount'] * pow(10, 10)
        stock_data.loc[stock_data['n_pctChange'] > 21, 'n_pctChange'] = 21
        stock_data['t_tradingDate'] = stock_data['t_tradingDate'].apply(lambda x: x.replace('-', ''))
        stock_data = stock_data.sort_values(['c_code', 't_tradingDate'])
    
        
        #逐日计算因子值
        factor_date = [x.replace('-', '') for x in TradeDate().get_trading_days(start_date,end_date)]
        factor = []
        for i in factor_date:
            s_date = TradeDate().get_previous_trade_dt(i, self.n)
            temp = stock_data[(stock_data['t_tradingDate']<=i) & (stock_data['t_tradingDate']>s_date)]
            
            # 剔除近20个交易日内有效数据不超过10天的个股
            date_count = temp.groupby('c_code').count()
            date_few = date_count[date_count['t_tradingDate'] <= 10].index
            temp = temp[~temp['c_code'].isin(date_few)]
            
            covariance = temp.groupby('c_code')[['n_pctChange', 'n_amount']].cov()
            covariance = covariance.reset_index()
            covariance = covariance[covariance['level_1'] == 'n_amount'][['c_code', 'n_pctChange']].set_index('c_code')
            
            factor.append(covariance['n_pctChange'].rename(i))
            print(i)
            
        result = pd.concat(factor,axis=1).T
        result = result.stack().rename('n_value').reset_index()
        result.columns = ['t_date','c_asset','n_value']
        result['t_date'] = result['t_date'].apply(lambda x: parse(x).strftime("%Y-%m-%d"))
        result = result.sort_values(by=['t_date','c_asset'])
        return result

def get_factor(param=None):
    factor = Liquidity_Illiquidity2(name='Liquidity_Illiquidity2',
                                    description='改进非流动性因子',
                                    category=FactorCategory.Liquidity)
    return factor

#%%测试
if __name__ == '__main__':
    start_date='20210907'
    end_date='20210908'
    
    factor = get_factor()
    a = factor.compute(start_date,end_date)
    #a.to_csv('Liquidity_Illiquidity2.csv')
