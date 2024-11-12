import pandas as pd
import numpy as np
from dateutil.parser import parse
import statsmodels.api as sm

from hs_sdk.factor.base_factor import StockFactor
from hs_sdk.utils.define import FactorCategory
from hs_sdk.datasource.hsbasicdata import EquityData,TradeDate

#%%因子计算
def regression(data):
    """
    对每日数据进行截面回归，并返回回归系数
    """
    data_add = sm.add_constant(data)
    y = data_add['n_pctChange']
    x = data_add[['const', 'pre_pctChange', 'sign_vol']]
    model = sm.OLS(y, x).fit()
    return round(model.params[2], 6)


class Liquidity_Gamma(StockFactor):
    '''
    Liquidity_Gamma(增强非流动性因子)
    计算方法：回溯过去30天的日收益率和交易额
             用每日收益率序列对前一天收益率、当日收益率的sign函数*交易额进行回归
             因子结果为收益率对后者的回归系数
    参考研报：中信建投证券《流动性因子系统解读与再增强》
    '''
    def __init__(self, name, category, description, n=20):
        self.name = name
        self.category = category
        self.description = description
        self.n = n
        
    def compute(self,start_date,end_date):
        #获取前一个月的个股股价数据、清洗异常值
        pre_date = TradeDate().get_previous_trade_dt(start_date, self.n + 10)
        stock_data = EquityData().get_stocks_price(pre_date, end_date, ['n_pctChange','n_amount'])
        stock_data.loc[stock_data['n_amount']==0,'n_pctChange'] = np.nan
        stock_data = stock_data.dropna()
        stock_data.loc[stock_data['n_pctChange'] > 21, 'n_pctChange'] = 21
        stock_data.loc[stock_data['n_pctChange'] < -21, 'n_pctChange'] = -21
        
        # 处理成可用来回归的形式
        stock_pre_pivot = stock_data.pivot('t_tradingDate', 'c_code', 'n_pctChange').shift(1).dropna(how = 'all')
        stock_data_pre = stock_pre_pivot.unstack().reset_index().rename(columns = {0:'pre_pctChange'})
        stock_data['n_amount'] = stock_data['n_amount'] / pow(10, 9)
        stock_data['posi'] = stock_data['n_pctChange'] > 0
        stock_data['nega'] = stock_data['n_pctChange'] < 0
        stock_data['sign'] = 0
        stock_data['sign'] = stock_data['posi'] * 1 + stock_data['nega'] * -1
        stock_data['sign_vol'] = stock_data['sign'] * stock_data['n_amount']
        stock_data = stock_data[['c_code', 't_tradingDate', 'n_pctChange', 'sign_vol']]
        stock_data = stock_data.sort_values(['c_code', 't_tradingDate'])
        regression_data = stock_data.merge(stock_data_pre, on = ['c_code', 't_tradingDate']).dropna()
        
        # 为了和factor_date保持同样字符串格式便于比较提取，后者是为了能在函数里使用
        regression_data['t_tradingDate'] = regression_data['t_tradingDate'].apply(lambda x: x.replace('-', ''))
        
        #逐日计算因子值
        factor_date = [x.replace('-', '') for x in TradeDate().get_trading_days(start_date,end_date)]
        factor = []
        for i in factor_date:
            s_date = TradeDate().get_previous_trade_dt(i, self.n)
            temp = regression_data[(regression_data['t_tradingDate']<=i) & (regression_data['t_tradingDate']>s_date)]
            
            # 剔除近20个交易日内有效数据不超过10天的个股
            date_count = temp.groupby('c_code').count()
            date_few = date_count[date_count['t_tradingDate'] <= 10].index
            temp = temp[~temp['c_code'].isin(date_few)]
            
            coef = temp.groupby('c_code').apply(lambda x:regression(x))
            
            factor.append(coef.rename(i))
            print(i)
            
        result = pd.concat(factor,axis=1).T
        result = result.stack().rename('n_value').reset_index()
        result.columns = ['t_date','c_asset','n_value']
        result['t_date'] = result['t_date'].apply(lambda x: parse(x).strftime("%Y-%m-%d"))
        result = result.sort_values(by=['t_date','c_asset'])
        return result

def get_factor(param=None):
    factor = Liquidity_Gamma(name='Liquidity_Gamma',
                             description='增强非流动性因子',
                             category=FactorCategory.Liquidity)
    return factor

#%%测试
if __name__ == '__main__':
    start_date='20210907'
    end_date='20210908'
    
    factor = get_factor()
    a = factor.compute(start_date,end_date)
    #a.to_csv('Liquidity_Gamma.csv')
