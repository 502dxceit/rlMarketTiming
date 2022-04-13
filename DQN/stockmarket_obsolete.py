# 1 acquire data from (various) sources
# 2 process data for env usage
# 3 store data into database, env only read data from database

#http://baostock.com/baostock/index.php/A%E8%82%A1K%E7%BA%BF%E6%95%B0%E6%8D%AE
import baostock as bs
import pandas as pd
import datetime
import random
import sqlite3

def Landmarks(df,MDPP_D = 5, MDPP_P = 0.05): # a closure

    def _mark(df):
        '''计算data的界标，并在self.data上做标记增加字段landmark
        [1] Perng C S, Wang H, Zhang S R, et al. Landmarks: a new model for similarity-based pattern querying in time series databases[C]// International Conference on Data Engineering, 2000. Proceedings. IEEE, 2000:33-42.
        href="http://citeseer.ist.psu.edu/viewdoc/download?doi=10.1.1.120.3361&rep=rep1&type=pdf"
        remove two adjacent points $(x_i,y_i) ,(x_{i+1},y_{i+1})$ if  $x_{i+1}-xi < D$ and $\frac{|y_{i+1}-y_i|}{(|y_i|+|y_{i+1|)/2}} <P$
        '''
        d = df.copy()
        d.close = d.close.astype("float")
        # print('points:%d '%d.count(),end="")
        # y-x>0 and y-z < 0 /   -     d.diff(1)>0 and d.diff(-1)<0
        # y-x<0 and y-z > 0 \   -     d.diff(1)<0 and d.diff(-1)>0
        # y-x>0 and y-z > 0 ^   high  d.diff(1)>0 and d.diff(-1)>0
        # y-x<0 and y-z < 0 v   low   d.diff(1)<0 and d.diff(-1)<0
        d['landmark'] = '-'
        d.loc[(d.close.diff(1)>0)&(d.close.diff(-1)<0),'landmark'] = '/'
        d.loc[(d.close.diff(1)<0)&(d.close.diff(-1)>0),'landmark'] = '\\'
        d.loc[(d.close.diff(1)>0)&(d.close.diff(-1)>0),'landmark'] = '^'
        d.loc[(d.close.diff(1)<0)&(d.close.diff(-1)<0),'landmark'] = 'v'
        d = d[d.landmark.isin(['^','v'])]
        for _ in range(MDPP_D):
            d = d[~(2*abs(d.close.diff(1))/(abs(d.close)+abs(d.close.shift(1)))<MDPP_P)] # 找出相邻点满足变化<P的数据行
        df['landmark'] = '-'
        df.loc[d[d.landmark == '^'].index,'landmark'] = '^'
        df.loc[d[d.landmark == 'v'].index,'landmark'] = 'v'
        return df

    def landmarks_after(datetime, n = 6):
        '''future n landmarks after the day '''
        lm = series[series.landmark.isin(['v','^'])] # series来自函数外部
        lm_after = lm[lm.time > datetime] #.copy() # .copy() makes sure not to modify the original data
        return lm_after[0:n] # at the end of stock series, there may be less than 6.

    series = df # df without .copy(), the df will be modified 
    _mark(series) # calculate and mark field 'landmark' with 'L' for low or 'H' for high
    return landmarks_after # returns a function for stockmarket to use
###################################
class StockMarket():
    # import pysnooper
    # @pysnooper.snoop()
    def __init__(self,begin='2006-01-01', howmany_samples = 10, more_online = True, local_db='stocks.db'):
        self.howmany_samples = howmany_samples
        self.begin = begin  # datetime.date(*map(lambda x:int(x), begin.split('-'))) instead for datetime type
        self.conn = sqlite3.connect(local_db)
        self.more_online = more_online # local first, if want more, go online
        bs.login(user_id="anonymous", password="123456")
        # try to load all tickers from local database if available, go online otherwise.
        try:
            local_tickers = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' and name like 'stock_%'",self.conn)
            local_tickers = local_tickers.name.apply(lambda x:x.replace('stock_','').replace("_",'.'))
            local_tickers = pd.read_sql_query('SELECT * FROM all_tickers where code in (\'%s\');'%'\',\''.join(list(local_tickers)),self.conn)
            remote_tickers = pd.DataFrame() # have to assign it before use it in else
            if len(local_tickers) < self.howmany_samples: # raise exception to go online for more
                raise Exception("lack of %d samples"%(self.howmany_samples-len(local_tickers))) 
        except Exception as e:
            print(e,", will go online, make sure Internet connection works ...")
            # self.all_tickers = bs.query_all_stock().get_data() # alternative to the following line 
            remote_tickers = bs.query_stock_basic().get_data()
            remote_tickers.to_sql("all_tickers",self.conn,if_exists="replace")
        more = max(0,self.howmany_samples-len(local_tickers))
        all_tickers = local_tickers.append(remote_tickers.sample(more),sort=False)

        # type = 1 for stocks|= 2 for indices
        # status = 0 for abnormal stocks, status = 1 for normal
        self.stocks = all_tickers[(all_tickers.type == '1') & (all_tickers.status =='1')] # 正常交易的股票
        self.indices = all_tickers[(all_tickers.type == '2') & (all_tickers.status =='1')] # 正常指数
        self.stocks_iter = self.stocks.sample(self.howmany_samples,replace=True).iterrows() 

    def __del__(self):
        # bs.logout() # don't need to logout manually?
        if self.conn: self.conn.close()

    def __save__(self):
        '''save to databases under update_or_insert policy'''
        self.stock.to_sql("stock",self.conn)
        
    def standardization(self):
        ''' to normalize data to [0,1], would be better?
        including action = buy, sell, confidence
        and state = open, close, low, high, volume/turn 
        + market_open, market_close, market_low, market_high, market_volume
        '''
        pass

    def minutely(self,ticker,ktype='5'):
        ''' try to load minutely data from local database if the ticker available, 
        go online otherwise.
        '''
        try:
            t = ticker.replace('.',"_") # sql does not allow '.' in table name
            self.stock  = pd.read_sql_query('SELECT * FROM stock_%s;'%t, self.conn)
        except Exception as e:
            print(e,", will go online for ticker %s data ... "%ticker)
            rs = bs.query_history_k_data_plus(ticker,
                "date,time,code,open,high,low,close,volume,adjustflag",
                start_date = self.begin,
                frequency = ktype, adjustflag="3")
            self.stock = rs.get_data() if rs.error_code == '0' else None
            self.stock[['open','high','low','close','volume']] = self.stock[['open','high','low','close','volume']].astype('float')
            self.stock.to_sql("stock_%s"%t,self.conn,if_exists="replace")
        return self.stock
    
    def next_ticker(self): # generator
        try:
            _, t = next(self.stocks_iter)
        except StopIteration:
           return "end of market" # how does main loop deal with this?
        print("working on {} {}".format(t.code, t.code_name))
        if not self.minutely(t.code).empty:
            self.landmarks = Landmarks(self.stock) 
        return t.code
        # a generator version, which may easily handle the StopInteration
        # for _,t in self.stocks.sample(self.howmany_samples,replace=True).iterrows():
        #     print("retrieving {} data ...".format(t.code_name))
        #     if not self.minutely(t.code).empty:
        #         self.current_oclh_iter = self.stock.iterrows()
        #         self.landmarks = Landmarks(self.stock)  
        #     done = yield self.stocks.code # curiously, yield doesn't work being invoked by next_oclh()
        #     if done: break # to received message from invoker as gen.send(msg)

    # as generator to yield result on each step
    # but iterator doesn't work since StopIteration is not captured until all rows have been popped.
    # This makes 'end_of_stock=True' returned one step after. In env.step(), however, end_of_stock should be returned
    # together with the last row. Therefore, we have to use 'if i==len(df)-1' to make it possible.
    def next_oclh(self):
        self.next_ticker() #for j, t in self.next_ticker(): # make next_ticker() and next_oclh() as a whole
        for i, row in self.stock.iterrows():
            end_of_stock = True if i == len(self.stock)-1 else False # last row of data
            lm = self._rest_oclh(row) # future several oclh from this time
            done = yield row, lm, end_of_stock  # after yield, hang here for further invoke, then goes to next line
            if done and not end_of_stock : break # to received message from invoker as gen.send(msg)
                
    def next_oclh_bn(self,batch_size=1):
        '''next batch of oclh, with normalization'''
        self.next_ticker() #for j, t in self.next_ticker(): # make next_ticker() and next_oclh() as a whole
        for i, row in self.stock.iterrows():
            end_of_stock = True if i == len(self.stock)-1 else False # last row of data
            lm = self._rest_oclh(row) # future several oclh from this time
            done = yield row, lm, end_of_stock  # after yield, hang here for further invoke, then goes to next line
            if done and not end_of_stock : break # to received message from invoker as gen.send(msg)

    def _rest_oclh(self, begin, end = None): # find low,high from begin to the rest of the day
        d = self.stock
        rest_oclh = d[(d.date == begin.date)&(d.time>begin.time)] # start from this time with this day
        rest_oclh = sorted(list(rest_oclh.low)+list(rest_oclh.high)+list(rest_oclh.open)+list(rest_oclh.close))
        # rest_oclh = set(rest_oclh) # reduce the duplicated, one way is not to do it bcz it reflects frequecy of trans
        return sorted(rest_oclh)

    def _future_dist(self,begin,end = None): # find future distribution from begin to the rest of the day
        d = self.stock
        rest_oclh = d[(d.date == begin.date)&(d.time>begin.time)]
        rest_oclh = rest_oclh[['open','high','low','close','volume']].astype('float')
        dist = sorted(list(rest_oclh.low)+list(rest_oclh.high)+list(rest_oclh.open)+list(rest_oclh.close))
        # use an uniform distribution to select low, how on each half, respectively
        # however, taking 'volume' into account, it is probably a distribution fairly away from uniform
        return random.choice(dist[:int(len(dist)/2)]), random.choice(dist[int(len(dist)/2):]) 

    def save_to(self, file = ''):
        ''' save current data to file
        - self.all_tickers
        - self.current_ticker
        - self.ticker_begin
        - self.ticker_end
        '''
        pass

    def load_from(self, file=''):
        pass

    def save(self):
        '''save to database'''
        pass

    def load(self):
        '''load from database'''
        pass

if __name__ == "__main__":
    sm = StockMarket()
    for n,lm,done in sm.next_oclh():
        print(n.code,n.date, n.time, n.close,lm)
        input("end of {}, continue?(Y/n):".format(n.code)) if done else None
        break # just one row for test