#coding=utf-8
import sys,os
sys.path.append(os.getcwd())
sys.path.append('..') # 需要引用父目录的data_work和utils
print(sys.path)
import gym
from gym import spaces
import numpy as np
import pandas as pd
from data_work import DataStorage
from utils import str2nparray # 加入绝对路径后，两个utils.py冲突了
from typing import List
import pysnooper

from globals import *

commission = 0.001 # 小于交易成本，投资没有意义
state_space = 14 # DQN.input_dim = len([*tech_indicator_list, *after_norm])
action_space = 3 # spaces.Box(low=np.array([0, 0]), high=np.array([10, 10]), dtype=np.float16) 
observation_space = spaces.Box(low=100, high=100, shape=(5,), dtype=np.float16)

class StockMarketEnv(gym.Env):
    """Customized Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df:pd.DataFrame = None)->None:
        super(StockMarketEnv, self).__init__()
        # 按照预处理结果，字段均在[-100,100]之间，如果action直接输出价格区间，也是[-100,100]比较合理
        # 如果action输出的是买卖动作，action_space in [-1,0,1]比较合理
        # self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([10, 10]), dtype=np.float16) #buy, sell, confidence
        self.action_space = spaces.Discrete(3)  # action 应该-1， 这应该写在agent里
        self.observation_space = spaces.Box(low=100, high=100, shape=(5,), dtype=np.float16) # input_dim = len([*tech_indicator_list, *after_norm])
        # observation_space意义何在
        
        self.ds = DataStorage()
        # pd.DataFrame(columns=["date","tic","open","close","high","low","volume",...,"embedding"]) 
        # note: 'embedding' is a string format vector as "25.3,23,67.0,23.1,62.12"
        self.df = self.ds.load_processed() if df is None else df
        self.window_size = 1    # 后面注意和globals.window_size 匹配
        self.state_space = (indicators).__len__() * self.window_size    #  + oclhva_after
        self.reset()
        

    def reward(self,action:int,day:str,price:float) -> float :
        '''  check the documentation for detail 
        '''
        extremums = self.df[self.df.landmark.isin(["v","^"])]
        # print(extremums)
        epsilon = 0.00005
        s = {-1:'^',0:'-',1:"v"}.get(action)  # [sell, hold, buy]
        # 针对第一个拐点之前和最后一个拐点之后的动作点设置分支，否则会序列越界
        # print(extremums.date, "-------------")
        if day < extremums.date.iloc[0]:
            y1 = self.df.iloc[0]
            y2, y3 = extremums[day <= extremums.date].iloc[0],extremums[day <= extremums.date].iloc[1]  #动作点 后一个，后二个拐点
        elif day > extremums.date.iloc[-1]:
            y1 = extremums[extremums.date <= day].iloc[-1]  # 动作点前一个拐点
            y2, y3 = self.df.iloc[-2], self.df.iloc[-1]     # ??
        elif day > extremums.date.iloc[-2]:
            y1 = extremums[extremums.date <= day].iloc[-1]  # 动作点前一个拐点
            y2 = extremums[day <= extremums.date].iloc[0]
            y3 = self.df.iloc[-2], self.df.iloc[-1]
        else:
            y1 = extremums[extremums.date <= day].iloc[-1]  # 动作点前一个拐点
            y2, y3 = extremums[day <= extremums.date].iloc[0],extremums[day <= extremums.date].iloc[1]  #动作点 后一个，后二个拐点
        # y1, y2 = y1, y2 if y1.landmark == s else y2, y3     # me :?
        y1, y2 = y1 if y1.landmark == s else y2, y2 if y1.landmark == s else y3
        # 按上述方法yi 是一个序列，下面得把yi中的close拿出来
        # yi总是出现变成tuple的情况？
        y1 = pd.Series(y1)
        y2 = pd.Series(y2)
        try:
            if action in [1,-1]: 
                # r_ = abs(y2.price - price) / (abs(price - y1.price) + epsilon) 
                r_ = abs(y2.close - price) / (abs(price - y1.close) + epsilon) 
            #   r = np.tanh(abs(y2.price - price) - abs(price - y1.price)) # alternative
            else : 
                # print(y2, type(y2))   y2可能变成tuple?
                
                
                    r_ = min(abs(price - y1.close), abs(price - y2.close))/(abs(price - (y1.close + y2.close)/2) + epsilon) 
        except:
                # 该字段y2只有一列数据date？
            return 0
        return np.tanh(np.log(r_)) 
    
    def step(self, action:int) : # -> np.array, float, bool, dict:
        ''' s_, r, done, info = step(a)
        a       list of 1 integer in (1,0,-1), namele [buy,hold,sell], for compatibility only
        s_      list              state   [datetime,o,c,l,h,v,t ...] 
        r       float             reward  from  _reward(a, landmarks)    
        done    False by default, True when it reachs the stock's end or run out of money
        info    {} anything you want to tell the rl_agent, for compatibility with gym-like env.
        '''
        state_, reward,done,info = None, 0, False,{}
        try:
            i, row = next(self.step_iter) # (index,[datetime,o,c,l,h,v,t ...], landmarks, episode done or not)
            done = True if i == len(self.df)-1 else False # is last row of data, don't try next lah.
            state_ = row[indicators] #  + oclhva_after str2nparray(row.embedding) #"2.3, 3.2, 1.6, 0.1, ..." -> '"np.array[2.3,3.2,1.6,0.1,...]"
            # print(row)
            reward = self.reward(action,day = row.date, price = row.close) 
            info = {}
        except StopIteration:
            print("end of dataset")
            done = True

        return state_, reward, done, info

    def reset(self):
        ''' switch to next ticker, and start from the first day
        '''
        # 原股票重新开始训练。在这里换一个股票会不会更好？
        self.step_iter = self.df.iterrows() # init an iterator   
        state0, _, _ ,_ = self.step(action=0)
        return state0

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    @classmethod
    def make(str = "StockMarket"): #to be compatible with gym
        pass

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