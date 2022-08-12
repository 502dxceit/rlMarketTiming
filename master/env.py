#coding=utf-8
import sys,os
sys.path.append(os.getcwd())
sys.path.append('..') # 需要引用父目录的data_work和utils
import gym
from gym import spaces
import numpy as np
import pandas as pd
from data_work import DataStorage, DataWorker
from utils import str2nparray # 加入绝对路径后，两个utils.py冲突了
from typing import List
import pysnooper
from preprocess import Preprocessor
from globals import *

commission = 0.001 # 小于交易成本，投资没有意义
state_space = 18 # DQN.input_dim = len([*tech_indicator_list, *after_norm])
action_space = 3 # spaces.Box(low=np.array([0, 0]), high=np.array([10, 10]), dtype=np.float16) 
observation_space = spaces.Box(low=100, high=100, shape=(5,), dtype=np.float16)

class StockMarketEnv(gym.Env):
    """Customized Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self)->None:
        super(StockMarketEnv, self).__init__()
        # 按照预处理结果，字段均在[-100,100]之间，如果action直接输出价格区间，也是[-100,100]比较合理
        # 如果action输出的是买卖动作，action_space in [-1,0,1]比较合理
        # self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([10, 10]), dtype=np.float16) #buy, sell, confidence
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=100, high=100, shape=(5,), dtype=np.float16) # input_dim = len([*tech_indicator_list, *after_norm])

        
        self.ds = DataStorage()
        # pd.DataFrame(columns=["date","tic","open","close","high","low","volume",...,"embedding"]) 
        # note: 'embedding' is a string format vector as "25.3,23,67.0,23.1,62.12"
        # self.df = self.ds.load_processed() if df is None else df
        # !!!! 使用env请用env.step初始化self.df
        self.window_size = 20    # 后面注意和globals.window_size 匹配
        self.state_space = (indicators + ['open_', 'close_', 'low_', 'high_', 'volume_', 'amount_', 'open_2', 'close_2', 'low_2', 'high_2', 'volume_2', 'amount_2']).__len__() * self.window_size    #  + oclhva_after


    def reward(self,action_:int,day:str,x:int) -> float :
        '''  check the documentation for detail 
        '''
        if self.rewardbase[self.rewardbase[:, 0]<x, 0].__len__() < 1 or self.rewardbase[self.rewardbase[:, 0]>x, 0].__len__() < 2:
            return 0

        action = action_ - 1
        # self.rewardbase
        epsilon = 0.00005

        x1 = self.rewardbase[self.rewardbase[:, 0]<x, 0][-1]
        x2, x3 = self.rewardbase[self.rewardbase[:, 0]>=x, 0][0], self.rewardbase[self.rewardbase[:, 0]>=x, 0][1]
        
        y1, y2, y3 = self.rewardbase[self.rewardbase[:, 0]==x1, 1], self.rewardbase[self.rewardbase[:, 0]==x2, 1], self.rewardbase[self.rewardbase[:, 0]==x3, 1]
        # 上面这行不够优雅，想了很久没有优雅的写法

        # rewardbase_sign 为True，则y1, y2, y3分别为v ^ v
        if y1 != self.y1_memory:
            self.rewardbase_sign = not self.rewardbase_sign
            self.y1_memory = y1

        y = self.y_(x, x1, y1, x2, y2)

        # if action == 1:
        #     y1, y2 == y2, y1

        if self.rewardbase_sign:
            # v ^ v
            if action == 1:
                r_ = abs(y2 - y) / (abs(y - y1) + epsilon)
            if action == -1:
                r_ = abs(y - y1) / (abs(y2 - y) + epsilon)
        else:
            # ^ v ^
            if action == 1:
                r_ = abs(y - y1) / (abs(y2 - y) + epsilon)
            if action == -1:
                 r_ = abs(y2 - y) / (abs(y - y1) + epsilon)


        # y1, y2 = y1, y2 if self.rewardbase_sign else y2, y3     # me :?
        if action == 0:
            r_ = min(abs(y - y1), abs(y - y2))/(abs(y - (y1 + y2)/2) + epsilon) 
            
        # if r_ == float("nan"):
        #     breakpoint()
        
        return np.tanh(np.log(r_))[0]


    
    def step(self, action:int) : # -> np.array, float, bool, dict:
        ''' s_, r, done, info = step(a)
        a       list of 1 integer in (1,0,-1), namele [buy,hold,sell], for compatibility only
        s_      list              state   [datetime,o,c,l,h,v,t ...] 
        r       float             reward  from  _reward(a, landmarks)    
        done    False by default, True when it reachs the stock's end or run out of money
        info    {} anything you want to tell the rl_agent, for compatibility with gym-like env.
        '''
        state_, reward ,done, info = None, 0, False,{}
        try:
            # i, row = next(self.step_iter) # (index,[datetime,o,c,l,h,v,t ...], landmarks, episode done or not)
            row = next(self.step_iter)  # 用rolling_windows做迭代器，返回row: dataframe
            # print(row, "---------" ,self.df.loc[row.index[-1], :])     # row.index[0]就是current x
            # done = True if i == len(self.df) - 1 else False # is last row of data, don't try next lah.
            state_columns = indicators + ['open_', 'close_', 'low_', 'high_', 'volume_', 'amount_', 'open_2', 'close_2', 'low_2', 'high_2', 'volume_2', 'amount_2']
            state_ = row.loc[:, state_columns].values.flatten() #  + oclhva_after str2nparray(row.embedding) #"2.3, 3.2, 1.6, 0.1, ..." -> '"np.array[2.3,3.2,1.6,0.1,...]"
            state_ = np.pad(state_, (self.state_space - state_.__len__(), 0), 'constant', constant_values=(0, 0))   # state不足长前面补位0， 因为rolling的前几项前面为空
            
            
            reward = self.reward(action, day = row.date.iloc[-1], x = row.index[-1]) 
            info = {}
        except StopIteration:
            state_ = [1]*self.state_space
            done = True
        return state_, reward, done, info

    def reset(self):
        ''' switch to next ticker, and start from the first day
        '''
        # 原股票重新开始训练。在这里换一个股票会不会更好？

        df, df_market = self.data_get()
        self.df = Preprocessor(df).bundle_process() # 每一个episode换一只股票
        self.df_market = Preprocessor(df_market).bundle_process(if_market=True)
        self.df_market.columns = ['code2', 'date2', 'open2', 'high2', 'low2', 'close2', 'pre_close2', 'pct_chg2',
                                    'volume2', 'amount2', 'open_2', 'close_2', 'low_2', 'high_2', 'volume_2',
                                    'amount_2']     # 防止连接字段冲突
        # 日期字段去掉 - ，转为int，否则无法pd.merge
        for i in range(self.df.__len__()):
            self.df.iloc[i, 1] = self.df.iloc[i, 1].replace("-", "")

        for i in range(self.df_market.__len__()):
            self.df_market.iloc[i, 1] = self.df_market.iloc[i, 1].replace("-", "")

        self.df["date"] = self.df["date"].astype("float64")
        self.df_market["date2"] = self.df_market["date2"].astype("float64")
        self.df = pd.merge(self.df, self.df_market, how="left", left_on="date", right_on="date2", sort=False)

        self.df = self.df.dropna()

        # self.step_iter = self.df.iterrows() # init an iterator   

        self.step_iter = self.df.rolling(self.window_size).__iter__()   # init an iterator with rolling window
        
        self.rewardbase, self.rewardbase_sign = self.reward_base()
        self.y1_memory = self.rewardbase[0, 1]  #监视x有没有超过x2，有则滚动一步
        
        state0, _, _ ,_ = self.step(action=0)
        return state0

    def data_get(self):
        # 获取不过短的数据表
        df, df_market, code = DataWorker().get()
        if df.__len__() < 1000:
            df, df_market = self.data_get()
        else:
            ...
        return df, df_market

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    @classmethod
    def make(str = "StockMarket"): #to be compatible with gym
        pass

    def y_(self, x, x1, y1, x2, y2):
        #计算直线上的中间点
        if x1<=x and x<x2:
            return (y2-y1)*(x-x1)/(x2-x1) + y1
        else:
            y_mirror = (y2-y1)*(x-x1)/(x2-x1) + y1
            if abs(x-x1)<abs(x-x2):
                return 2*y1-y_mirror
            else:
                return 2*y2-y_mirror

    def reward_base(self):
        # sample a numpy(n, 2) for reward caculate, [x, close]
        ay = np.expand_dims(self.df.iloc[0, :].close, axis=0)   # 这个参数里加.values就会报错，为何这个的.close出来的是numpy.float64?因为是只有一个值？
        by = self.df[self.df.landmark.isin(["v","^"])].close.values
        cy = self.df.iloc[[-2, -1], :].close.values

        a = np.expand_dims(self.df.index[0], axis=0)
        b = self.df[self.df.landmark.isin(["v","^"])].index.values
        c =  self.df.iloc[[-2, -1], :].index.values

        # print(cy.shape, ay.shape, by.shape)
        # print(type(ay), type(by), type(cy))
        # print(c.shape, a.shape, b.shape)
        # print(type(a), type(b), type(c))
        
        x = np.expand_dims(np.concatenate([a, b, c], axis=0), axis=1) 
        
        y = np.expand_dims(np.concatenate([ay, by, cy], axis=0), axis=1)
        
        # print(x.shape, y.shape)
        res = np.concatenate([x, y], axis=1)

        ### 判断标记点是以v起始或以^起始 ###
        # print(self.df[self.df.landmark.isin(["v","^"])].landmark.values[0] )    # pandas真是朵奇葩
        if self.df[self.df.landmark.isin(["v","^"])].landmark.values[0] == "^":
            sign = True
        else:
            sign = False

        ### sign = True，剩余序列第一个标点是v 反之亦然 ### 动态更新，指明y1


        return res, sign

if  __name__ == "__main__":
    env = StockMarketEnv()
    s = env.reset() # episode starts here
    var = 0.1
    for t in range(5):
        a = np.clip(np.random.normal(a,var),0.,2) # a = agent(s)
        s_, r, done, info = env.step(a)
        print("t={},s= {},\ta={},\tr = {},\ts_={}".format(t,s,a,r,s_))
        env.render()
        s = s_

    
