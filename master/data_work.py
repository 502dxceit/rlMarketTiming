from subprocess import call
import tushare as ts
import datetime
import os
import sqlite3
import pandas as pd
from functools import partial
from globals import MAIN_PATH   # 
import pysnooper
from retrying import retry
import baostock
import time

DATABASE = "stock.db"
DATABASE_PATH = MAIN_PATH  # 这句话在其他import的时候就已经执行，所以未必能达到想要的效果 # 直接调用上面globals.MAIN_PATH 导致不能正确建立conn，main不能及时修改globals.MAIN_PATH
DATABASE_PATH = os.getcwd()     # 解决方法
# print(DATABASE_PATH + "-------------------------------")
RAW_DATA = "downloaded" # by 'date'
PROCESSED_DATA = "downloaded" # by 'date'
TRAINED_DATA = "downlowded_tarined" #  by 'date'
EVALUATED_DATA = "downloaded" # by 'date' 下一回合数据到来的时候即清空
TRAIN_HISTORY_DATA = "train_history" # by 'episode'  长期保存做评估
ACTION_HISTORY_DATA = "action_hisory" #  by 'action' action_history结构和evaluated一样，但去掉了action==0的记录，供长期保存做评估
PREDICTED_DATA = "predicted" # by 'action', action prediction for WatchList，保存用作评估
MODEL_PATH = "./model/" # table name
MODEL_NAME = "trained_model"


class DataStorage():
    def __init__(self, path = DATABASE_PATH , database = DATABASE) -> None:
        self.conn = sqlite3.connect(os.path.join(path, database))
        # raw, preprocessed, trained, evaluated can be unified as in one table, or two
        # since they are all by 'ticker,date'
        # the following wrappers follows the order of whole process
        # raw.columns = baostock.columns = ['date', 'code', 'open', 'high', 'low', 'close', 
        # 'preclose', 'volume', 'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'isST']
        # preprocessed.columns += [normalized oclhva,landmarks,indicators,embeddings]
        # trained.columns = ['date','ticker','landmark','close','action','reward']
        # evaluated.columns += ['asset_pct_chg'] 
        self.save_raw = partial(self.save,to_table = RAW_DATA)
        self.load_raw = partial(self.load,from_table = RAW_DATA)
        self.save_processed = partial(self.save,to_table = PROCESSED_DATA)
        self.load_processed = partial(self.load, from_table = PROCESSED_DATA)
        self.save_trained = partial(self.save,to_table = TRAINED_DATA)
        self.load_trained = partial(self.load,from_table = TRAINED_DATA)
        self.save_evaluated = partial(self.save, to_table = EVALUATED_DATA)
        self.load_evaluated = partial(self.load, from_table = EVALUATED_DATA)
        # 'train_history' is to preserve the history of training, by 'episode'
        # train_history.columns =  ['episode','ticker','train_date','mean','std','asset_change']
        self.append_train_history = partial(self.save, to_table = TRAIN_HISTORY_DATA,if_exists='append')
        self.load_train_history = partial(self.load, from_table = TRAIN_HISTORY_DATA) 
        # 'action_history' is to preserve the history of predicted action for train data, by 'action'
        # action_history.columns =  ['date','ticker','action','reward','asset_change'] 
        self.append_action_history = partial(self.save, to_table = ACTION_HISTORY_DATA,if_exists='append')
        self.load_action_history = partial(self.load, from_table = ACTION_HISTORY_DATA) 
        # 'predicted' is to preserve the history of predicted action for WatchList, by 'action'
        # predicted.columns = ['predict_date','ticker','action','asset_pct_chg']
        self.append_predicted = partial(self.save, to_table = PREDICTED_DATA,if_exists='append')
        self.load_predicted = partial(self.load, from_table = PREDICTED_DATA)

    def __del__(self) -> None:
        self.conn.close()

    def load(self,from_table:str) -> pd.DataFrame:
        return pd.read_sql('SELECT * FROM %s'% from_table, con = self.conn,index_col='index')

    def save(self, df:pd.DataFrame, to_table:str, if_exists='replace'):
        return df.to_sql(name=to_table,con = self.conn, if_exists = if_exists)
    
    # def save_raw(self,df:pd.DataFrame):
    #     return self.save(df, RAW_DATA)
    # self.save_raw = lambda df: self.save(df,RAW_DATA)
    # def load_raw(self):
    #     return self.load(RAW_DATA)
    # def save_processed(self,df:pd.DataFrame):
    #     return self.save(df, PROCESSED_DATA)
    # def load_processed(self):
    #     return self.load(PROCESSED_DATA)
    # def save_trained(self,df:pd.DataFrame):
    #     return self.save(df, TRAINED_DATA)
    # def load_trained(self):
    #     return self.load(TRAINED_DATA)
    # def save_evaluated(self,df:pd.DataFrame):
    #     return self.save(df, EVALUATED_DATA)
    # def load_evaluated(self):
    #     return self.load(EVALUATED_DATA)
    # def save_predicted(self,df:pd.DataFrame,if_exists='append'):
    #     return self.save(df, PREDICTED_DATA)
    # def load_predicted(self):
    #     return self.load(PREDICTED_DATA)

class DataWorker(object):
    def __init__(self) -> None:
        super().__init__()
        tushare_token = "c576df5b626df4f37c30bae84520d70c7945a394d7ee274ef2685444"
        # tushare_token = "72d1e47c3b0728a26bfc4a9f54132b195890fa843815f896708515f1" # 294694@tushare.pro的token
        ts.set_token(tushare_token)
        self.pro = ts.pro_api()
        

    @property
    def all_tickers(self):
        return self.pro.stock_basic(exchange='', list_status='L', fields='ts_code, name, area,industry,list_date')

    def market_of(self,ticker:str)->str:
        ''' get market name of the same start and end date where the stock belongs to
        '''
        index_codes = ['000001', '000300', '399001', '399005', '399006']
        hk_index_codes = []
        us_index_codes = []
        s = {'SH':'000001.SH','SZ':'399001.SZ','BJ':'883991.BJ'}
        s = {'SH':'sh.000001','SZ':'sz.399001','BJ':'bj.883991'}        # baostock风格
        return s[ticker[-2:]]

    def get_hk(self,ticker:str = None, days_back:int = None, end = datetime.datetime.now().strftime("%Y%m%d"))-> pd.DataFrame: 
        return self.pro.hk_daily(ts_code = ticker,start_date = days_back ,end_date = end)

    def get_us(self,ticker:str = None, days_back:int = None, end = datetime.datetime.now().strftime("%Y%m%d"))-> pd.DataFrame: 
        return self.pro.us_daily(ts_code = ticker,start_date = days_back ,end_date = end)
    
    def get_oversea(self,**kwarg)->list[str]:
        '''according to 'exchange' in kwarg, determines the func to revoke''' 
        s = {'HK':self.pro.hk_daily, 'US':self.pro.us_daily}
        return s[exchange]

    @retry
    def get(self,ticker:str = None, days_back:int = None, end = datetime.datetime.now().strftime("%Y%m%d"), start = None)-> pd.DataFrame: 
        # retrieve all daily w.r.t. ticker from its list_date to now()
        # return ts.pro_bar(ts_code=ticker, adj='qfq', freq="d", start_date=ticker.list_date, end_date=datetime.datetime.now())
        # return self.pro.daily(ts_code = ticker.ts_code, start_date=ticker.list_date, end_date=datetime.datetime.now().strftime("%Y%m%d")
        code = ticker if ticker else self.all_tickers.sample(1).iloc[0].ts_code # retrieved was a set of records, needs a iloc[0] to take out the exact one.
        if start:
            ... 
        else:
            start = datetime.datetime.now() - datetime.timedelta(days=days_back) if days_back else '20000101'
        start = start.strftime("%Y%m%d") if days_back else '20000101'
        stock = self.pro.daily(ts_code = code, start_date = start, end_date = end)
        market = self.get_market(code, start = self.tp(stock.iloc[-1, 1]), end = self.tp(stock.iloc[0, 1]))     #baostock最后一天是开区间导致获取不了，所以stock最后一天不要了以匹配长度
        market = market.iloc[::-1].reset_index().drop(["index"], axis=1)
        # market 的 date 删掉 -
        for i in market.index:
            market.loc[i, "date"] = market.loc[i, "date"].replace("-", "") 
        market.rename(columns = {"preclose": "pre_close", "pctChg":"pct_chg"}, inplace=True)
        for i in ["open","high","low","close","pre_close","pct_chg","volume","amount"]:
            market.loc[:, i] = pd.to_numeric(market.loc[:, i])
        return stock[:-1], market, code
    def tp(self, str):
        # 把日期处理为baostock的格式，输入输出都是str
        assert str.__len__() == 8
        return str[0:4] + "-" + str[4:6] + "-" + str[6:]

    def get_market(self, code, start, end):
        
        rs = baostock.query_history_k_data_plus(self.market_of(code), "code,date,open,high,low,close,preclose,pctChg,volume,amount", start_date=start, end_date=end, frequency="d")

        return rs.get_data()

if __name__ == "__main__":
    dw = DataWorker()
    
    print(dw.all_tickers.sample(1).iloc[0].ts_code)



