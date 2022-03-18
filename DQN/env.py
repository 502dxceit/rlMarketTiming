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

commission = 0.001 # 小于交易成本，投资没有意义
state_space = 11 # DQN.input_dim = len([*tech_indicator_list, *after_norm])
action_space = spaces.Box(low=np.array([0, 0]), high=np.array([10, 10]), dtype=np.float16) 
observation_space = spaces.Box(low=100, high=100, shape=(5,), dtype=np.float16)

class StockMarketEnv(gym.Env):
    """Customized Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df:pd.DataFrame = None)->None:
        super(StockMarketEnv, self).__init__()
        # 按照预处理结果，字段均在[-100,100]之间，如果action直接输出价格区间，也是[-100,100]比较合理
        # 如果action输出的是买卖动作，action_space in [-1,0,1]比较合理
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([10, 10]), dtype=np.float16) #buy, sell, confidence
        self.observation_space = spaces.Box(low=100, high=100, shape=(5,), dtype=np.float16) # input_dim = len([*tech_indicator_list, *after_norm])
        self.ds = DataStorage()
        # pd.DataFrame(columns=["date","tic","open","close","high","low","volume",...,"embedding"]) 
        # note: 'embedding' is a string format vector as "25.3,23,67.0,23.1,62.12"
        self.df = self.ds.load_processed() if df is None else df
        self.reset()
  
    def reward(self,action:int,day:str,price:float) -> float :
        '''  check the documentation for detail 
        '''
        extremums = self.df[self.df.landmark.isin(["v","^"])]
        epsilon = 0.00005
        s = {-1:'^',0:'-',1:"v"}.get(action)  # [sell, hold, buy]
        y1 = extremums[extremums.date <= day].iloc[-1]  
        y2, y3 = extremums[day <= extremums.date].iloc[0],extremums[day <= extremums.date].iloc[1]
        y1, y2 = y1, y2 if y1.landmark == s else y2, y3 
        if action in [1,-1]: 
            r_ = abs(y2.price - price) / (abs(price - y1.price) + epsilon) 
        #   r = np.tanh(abs(y2.price - price) - abs(price - y1.price)) # alternative
        else : 
            r_ = min(abs(price-y1),abs(price-y2))/(abs(price-(y1+y2)/2)+epsilon) 
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
            state_ = str2nparray(row.embedding) #"2.3, 3.2, 1.6, 0.1, ..." -> '"np.array[2.3,3.2,1.6,0.1,...]"
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