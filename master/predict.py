'''
run preditor will return all the predicted action for tickers in watchlist 
'''
import datetime
import pandas as pd
from master.data_work import MODEL_PATH
from preprocess import Preprocessor 
from data_work import DataStorage, DataWorker
from env import state_space, action_space
from utils import str2nparray
from retrying import retry
import pysnooper
import globals

WatchList = [
    '000555.SZ',
    '601099.SH'
]

MODEL_PATH = ""


class Predictor:

    def __init__(self,model_path = MODEL_PATH,watch_list=WatchList):
        ...
        # self.agent = DQN(state_space, action_space, **config)   # .target_net
        # self.agent.load(model_path)
        # self.watchlist = watch_list if watch_list else self.load_watchlist() 
        # self.ds = DataStorage()
        # self.dw = DataWorker()
        # self.end_time = datetime.datetime.now().strftime('%Y-%m-%d')
        # self.days_back = 1000   # 某些股票最近没有数据，导致这项过少使获取的股票序列过短，没有拐点而报错
    
    def load_watchlist(self):
        '''
        return self.ds.load_watchlist() # later
        '''
        return WatchList

    def get_data(self,ticker:str):
        ''' get data of a ticker, then preprocess()
            we use days_back to control the length
        '''
        df = self.dw.get(ticker, days_back = self.days_back, end = self.end_time)
        return Preprocessor(df=df).bundle_process() 

    def predict(self, state):
        '''given a (latest) state, yields an action
            这个函数和DQN.predict() 似乎不统一，DQN.predict()返回的本来就是action啊！待调试
        '''
        action = self.agent.predict(state)    # 这一步直接返回预测值了最大Q值
        # print(q_values)
        # return self.agent.predict(state) # 应该这样就可以了的?
        # action = q_values.tolist().index(q_values.max()) - 1 # 直接返回[-1,0,1] 的值不就行了？
        # action_map = {-1:'sell',0:'hold',1:'buy'}
        # action = action_map[action]
        return action 

    def get_state(self,ticker:str):
        ''' given a ticker, returns a series of 'state's, as np.array
        '''
        df = self.get_data(ticker = ticker)
        # ss = [str2nparray(e) for e in df.embedding] # "1.1,3.0,6.2,3.8" => np.array([1.1,3.0,6.2,3.8]) # now without embedding
        ss = df.loc[df.index[-1], globals.indicators]
        return ss # return last embedding tensor as state

    def watch(self,watchlist=WatchList)->list[tuple]:
        ''' prediction for tickers in watchlist with their latest states
            returns a list of tuples or, returns a pd.Dataframe
            this function can be replace by latest_actions(·) 
        '''
        result = [(self.end_time, t,self.predict(self.get_state(t))) for t in watchlist]
        df = pd.DataFrame(result,columns=['date','ticker','action'],dtype=int)
        self.ds.append_predicted(df[df.action.isin([-1,1])], if_exists = 'append') # save only action in [-1,1]
        return result # or, df
    
    def trend(self,ticker:str)->pd.DataFrame:
        '''note: sliding window会在前window_size-1条记录产生空值，但在preprocessor.embedding()里已经去掉了这些空值的行
        '''
        df = self.get_data(ticker)
        ss = [str2nparray(e) for e in df.embedding] # "1.1,3.0,6.2,3.8" => np.array([1.1,3.0,6.2,3.8])
        df['action'] = [self.predict(s) for s in ss]
        return df

    def trends(self, watchlist=WatchList)->list[pd.DataFrame]:
        '''get 6 months trend of the stocks in watchlist'''
        # return list(map(lambda x:self.get_data,watchlist))
        return [self.trend(t) for t in watchlist]  

    def latest_actions(self,watchlist=WatchList)->list[tuple]:
        ''' pretty much the same as 'watch(·)'
            w.r.t. each ticker in watchlist, get the trend(t). latest action is the last row of the dataframe
            this func can also be replaced by:
                result = [(self.end_time, t, t.iloc[-1].action) for t in self.trends(WatchList)]
                df = pd.DataFrame(result,columns=['date','ticker','action'],dtype=int)
        '''
        latest_action = lambda t: self.trend(t).iloc[-1].action
        result = [(self.end_time, t,latest_action(t)) for t in watchlist]
        df = pd.DataFrame(result,columns=['date','ticker','action'],dtype=int)
        self.ds.save_predicted(df[df.action.isin([-1,1])], if_exists = 'append') # save only action in [-1,1]
        return result # or, df as 'st.table(df)' in visualize.py
        
if __name__ == "__main__":
    predictor = Predictor(MODEL_PATH, WatchList)
    print(predictor.latest_actions(WatchList))
    print(predictor.trends(WatchList))