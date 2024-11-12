import time

import pandas as pd
from hs_sdk.datasource.hsbasicdata import EquityData, TradeDate
from hs_sdk.factor.base_factor import StockFactor
from hs_sdk.utils.define import FactorCategory


# %%
class Reversal_DayExtremeReturn(StockFactor):
    """
    日内极端收益反转因子
    1.日内分钟收益：计算每天个股的日内分钟收益（从开盘后算起，故每天共239个分钟收益）
    2.个股分钟收益偏差：个股的分钟收益 - 该个股当日分钟收益中位数，取绝对值
    3.极端分钟收益：个股分钟收益偏差最大的一分钟的实际收益率（包含正负）
    4.对个股前20天（包括当天）的极端分钟收益取均值平滑
    参考研报：开源金工《市场微观结构研究系列（17）——日内极端收益前后的反转特性与因子构建》
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
        
        self.n = n

    def compute_one_day(self, data, date):
        """
        计算一天的因子值，传入数据为当天的所有分钟频数据
        :param data_group: all ['c_code','t_tradingDate','t_tradingTime','n_close',...]
        :param date: 'yyyy-mm-dd'
        :return: ['t_date','c_asset','n_value'] DataFrame
        """
        print(date)
        
        data_pivot = data.pivot(index = 't_tradingTime', columns = 'c_code', values = 'n_close')
        data_return = data_pivot.pct_change(1).dropna(how = 'all') * 100
        return_median = data_return.median()
        absolute_return = abs(data_return - return_median)
        extreme_index = absolute_return.apply(lambda x:x.argmax())
        extreme_return = data_return.apply(lambda x:x.iloc[extreme_index[x.name]])

        result = pd.DataFrame(extreme_return).reset_index().rename(columns = {'c_code':'c_asset', 0:'n_value'})
        result['t_date'] = date

        print(date, ' computed!')
        return result

    @staticmethod
    def look_back(factor: pd.DataFrame, n, m = 1):
        """
        以向前n天的平均值平滑当天的值
        :param factor: ['t_date','c_asset','n_value']
        :param n:
        :param m: min_periods，default 1
        :return: ['t_date','c_asset','n_value']
        """
        factor_pivot = factor.pivot(index='t_date', columns='c_asset', values='n_value').sort_index(ascending=True)
        factor_rolling = factor_pivot.rolling(n, n).mean().dropna(how = 'all')
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
        try:
            beg_date = TradeDate.get_previous_trade_dt(start_date, self.n - 1)
        except AssertionError:
            beg_date = start_date
        trade_list = TradeDate.get_trading_days(beg_date, end_date)
        if not trade_list:
            print("None")
            return None
        beg_index = 0  # changing
        finish_index = len(trade_list) - 1  # fixed

        df = []
        while beg_index <= finish_index:
            fields = ['n_close']
            date = trade_list[beg_index]
            data = EquityData.get_stocks_marketminute(date.replace('-', ''), date.replace('-', ''), fields)
            data.drop_duplicates(keep='first', inplace = True)
            print(
                f'current_date: {date}\t Minute data fetched, using {time.time() - self.local_time:.5f}s')
            loop_df = self.compute_one_day(data, date)
            df.append(loop_df)
            beg_index += 1
        df = pd.concat(df, ignore_index=True)
        df = df[['t_date', 'c_asset', 'n_value']]

        modified_df = self.look_back(df, self.n, self.n)
        return modified_df.sort_values(by=['t_date', 'c_asset'])


# %%
def get_factor(param=None):
    factor = Reversal_DayExtremeReturn(name='Reversal_DayExtremeReturn', 
                                       category=FactorCategory.Reversal,
                                       description='日内极端收益反转因子',
                                       n = 1)
    return factor


# %%
if __name__ == '__main__':
    fac = get_factor()
    start_date = '20201201'
    end_date = '20221206'
    fac_value = fac.compute(start_date, end_date)
    print(fac_value)
    fac_value.to_pickle('日内极端收益反转因子.pkl')
