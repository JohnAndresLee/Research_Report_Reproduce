from joblib import Parallel, delayed

import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
import multiprocessing

from hs_sdk.factor.base_factor import StockFactor
from hs_sdk.utils.define import FactorCategory
from hs_sdk.datasource.hsbasicdata import EquityData,TradeDate
#%%
def get_ma(MA,data):
    '''
    Parameters
    ----------
    MA : 需要计算的EMA
    data : 计算EMA用到的回看数据

    Returns
    -------
    df : pd.DataFrame
        单只股票的ma
    '''
    w = [np.log(.5)/np.log((x-1)/(x+1)) for x in MA]
    dic = dict(zip(MA,w))
    col = ['ma_'+str(x) for x in MA]
    df = pd.DataFrame(index = data.set_index(['t_tradingDate','c_code']).iloc[-1:].index,columns=col)
    for i,j in dic.items():
        weight = half_life(len(data),j)
        if len(data) < 252:
            return
        else:
            ma = (data['n_adjClose'].values * weight.values).sum()
            df['ma_'+str(i)] = ma
    return df

def regression(data):
    """
    对每日数据进行截面回归，并返回回归系数

    """
    data_add = sm.add_constant(data[['ma_3','ma_5','ma_10','ma_30','ma_60','ma_120','ma_240']].dropna())
    model = sm.OLS(data['n_pctChange']/100,data_add).fit()
    return model.params[1:]

def half_life(length,half_life):
    """半衰期参数计算
    parameters:
    -------------
    length: 时间长度
    half-life : 半衰期时间长度，即经过half-life后，系数衰减为原来0.5
    
    returns:
    --------------
    return a series
    """
    ind = pd.Series(range(length))-length+1
    res = ind.apply(lambda x:math.pow(2,x/half_life))
    res = res/res.sum()  #归一化处理
    return res

def applyParallel(dfGrouped,func,MA):
    '''
    代替 pd.groupby.apply()实现加速

    Parameters
    ----------
    dfGrouped :  pd.groupby
    func :  function
        想要 apply 的函数.

    Returns
    -------
    fac_single_day :  pd.DataFrame
        单日的截面因子值.

    '''
    ma_single_stock_one_day = Parallel(n_jobs = multiprocessing.cpu_count())(delayed(func)(MA,group) for _, group in dfGrouped)
    ma_one_day = pd.concat([x for x in ma_single_stock_one_day])
    return ma_one_day
#%%
class PV_SyntheticMovingAverage(StockFactor):
    """
    SyntheticMovingAverage (综合移动平均线)
    国泰君安证券“综合期限多样性的趋势选股策略”数量化专题之九十
    
    计算方法: 
    回归系数：将每日收益率与2日前各日级均线数据线性回归，得到每日的回归系数
    对回归系数进行平滑处理，即对过去22日取均值（不包括当日），得到当日回归系数
    再用当日的回归系数与均线预测后天的收益率，并以此作为因子值
    """
    def __init__(self, name, category, description,n):
        self.n = max(n,10)   #回溯天数，不能小于10天
        super().__init__(name, category, description)
    def compute(self, start_date, end_date):
        #获取日期及初始化存储相关信息的df
        new_start_date = TradeDate.get_previous_trade_dt(start_date,self.n+252)
        stock_price = EquityData.get_stocks_price(start_date = new_start_date,end_date = end_date,fields = ['n_adjClose','n_pctChange'])
        stock_price = stock_price.sort_values(by=['t_tradingDate','c_code'])
        
        tradedates = TradeDate.get_trading_days(new_start_date.replace('-',''), end_date)
        ma_df = pd.DataFrame()        
        #计算各日EMA均线 ma_n
        MA = [3,5,10,30,60,120,240]
        print('EMA Calculation Starts')
        for date in tradedates[:-251]:
            enddate = [x for x in tradedates if x>=date][251]
            print(enddate)
            temp = stock_price[(stock_price['t_tradingDate'] >= date) & (stock_price['t_tradingDate'] <= enddate)]
            df = applyParallel(temp.groupby('c_code'),get_ma,MA).sort_index()
            ma_df = pd.concat([ma_df,df])
            
        # ma_df 就是当日（结束时）的当日均线
        data = stock_price.merge(ma_df, on=['t_tradingDate','c_code'])
        
        for i in MA:
            data['ma_'+str(i)] = data['ma_'+str(i)]/data['n_adjClose']
        data = data.dropna().set_index(['t_tradingDate','c_code'])
        print(data)
        '''
        对于第t天，用t-2的MA回归第t天的收益率，得到回归系数
        回归系数做平滑
        再用回归系数 * 第t天的MA，预测t+2的收益率
        作为明日收盘是否需要买入的依据
        '''
        # 先保存所有的MA数据、以免因shift丢失
        ma_df = data.iloc[:,2:]
        
        data.iloc[:,2:] = data.iloc[:,2:].groupby('c_code').shift(2)   # 将MA下移2天
        data = data.dropna()
        coef = data.groupby('t_tradingDate').apply(lambda x:regression(x))
        coef = coef.rolling(self.n).mean()
        coef = coef.dropna()
        #用平滑的系数和当期均线组合预测下期收益
        ma_df['pre_return'] = 0
        ma_df = ma_df.reset_index()
        ma_df = ma_df[ma_df['t_tradingDate'].isin(coef.index)]
        ma_df = ma_df.reset_index().drop('index', axis = 1)
        result = pd.DataFrame()
        for name, group in ma_df.groupby('t_tradingDate'):
            temp = group.copy()
            temp['pre_return'] = (group[['ma_'+str(i) for i in MA]] * coef.loc[name]).sum(axis=1)
            result = pd.concat([result,temp])
        result = result[['t_tradingDate','c_code','pre_return']]
        result.columns = ['t_date','c_asset','n_value']
        return result

def get_factor(param=None):
    factor = PV_SyntheticMovingAverage(
                name='PV_SyntheticMovingAverage',
                description='综合期限均线因子',
                category = FactorCategory.PV,
                n = 22
                )
    return factor
#%%
if __name__ == '__main__':
    start_date = '20201117'
    end_date = '20221114'
    final = get_factor().compute(start_date,end_date)
    