import warnings
import numpy as np
import pandas as pd
import datetime
from data_work import DataStorage
from functools import reduce
import warnings
import pysnooper

# from main import train    

class Evaluator():
    '''
        dataworker获取raw的部分；preprocessor添加了一部分如landmark和若干技术指标用于embedding；trainer添加了['action','reward']
        需要评估的内容有：
        - reward 在一个股票学习中是不是呈总体上升趋势。在历史以来的学习中，reward有无上升趋势？这可能需要另外有个表保存 
        - action 与 landmark的重合度如何？ optional
        - 假设有100%资产，按照action操作，资产变化情况如何？增加一个字段d.asset_pctChg
        注：reward的设计，已经充分体现了动作的精度：越接近同买同卖点分数越高，因此不再需要分类指标来表达
    '''
    def __init__(self, trained = None, episode = None):
        self.episode = episode  # episode from train process
        self.ds = DataStorage()
        # before: trained.columns = ['date','ticker','landmark','close','action','reward']
        # after: evaluated.columns = ['date','ticker','landmark','close','action','reward','asset_pct_chg'] 
        self.trained = self.ds.load_trained() if trained is None else trained
        # train_history.columns =  ['episode','ticker','train_date','mean','std','asset_change']
        
        try:
            self.train_history = self.ds.load_train_history() 
            self.save_train_history()
        # action_history.columns =  ['date','ticker','action','reward','asset_change'] 
            self.action_history = self.ds.load_action_history() 
        # predicted.columns = ['predict_date','ticker','action','asset_pct_chg']
            self.predicted = self.ds.load_predicted() # columns=['date','ticker','action'] 
        except:
            warnings.warn("嗯？之前没有存过train_history？")
        
        self.trained = self.trained.update(self.asset_change(do_short=False),join='left', overwrite=True) 
        # 或者基于index合并，right_index = True
    
    def asset_change(self, do_short = False) -> pd.DataFrame:
        '''
            calculate asset change by percent at each action
            do_short = True if considering short operation else False. when action == -1 (sell)
            'short operation' means to operate reversely (borrow shares to sell then buy to return at lower price)
        '''
        self.trained[['asset_pct_chg']] = 0 # 第一次添加字段需要这样写
        actioned = self.trained[self.trained.action.isin([-1,1])] # 只选择有买卖动作的记录
        actioned.asset_pct_chg = actioned.close.pct_change() # 相比上一次的动作价格的变化比例
        actioned.dropna()  # pct_change()完第一行是空值
        # 这里逻辑可能还有问题：卖出时资产有增减，而买入资产则无增减，所以是否应该考虑r.action==1
        if do_short:
            alter_ =lambda i,r: -r.asset_pct_chg if r.action == -1 else r.asset_pct_chg 
        else:
            alter_ =lambda i,r: 0 if r.action == -1 else r.asset_pct_chg  # just sell, don't do short

        actioned.asset_pct_chg = [alter_(i,r) for i,r in actioned.iterrows()]

        return actioned[['date','asset_pct_chg']]
    
    def save_evaluated(self) -> bool:
        return self.ds.save_evaluated(self.trained) 

    def append_action_history(self) -> bool:
        ''' note: action_history does not resemble self.df loaded from trained, 'date' in consecutive manner
            instead, ds.save_actioned() only reserves the action in [-1,1] and its asset returns
            and ds.save_evaluated() the mean/std reward/per episode
        '''
        return self.ds.append_action_history(self.asset_change())

    def asset_change_eps(self) -> float:
        '''calculate asset change by percent after the whole episode'''
        return reduce(lambda a,b:a*(1+b),self.asset_change().asset_pct_chg,1)   # 累乘

    
    def save_train_history(self):
        '''note: df does not resemble self.df loaded from trained, 'date' in consecutive manner
            instead, ds.save_actioned() only reserves the action in [-1,1] and its asset returns
            and ds.save_evaluated() the mean/std reward/per episode
        '''
        
        # train_history ['episode','ticker','train_date','count',	'mean',	'std',	'min',	'25%',	'50%',	'75%','max','asset_pct_chg']
        df = self.trained[['reward']].describe().T  #注意这里必须两个方括号'[['才是dataframe，self.trained.reward是series，更不行

        df['date'] = datetime.datetime.now().strftime("%Y-%m-%d") # 增加当天日期字段

        # 不存在episode字段会报错
        df['episode'] = self.episode  # 增加episode字段，可以考虑不要这个字段，因为是append进表的，表的index就是递增
        df['ticker'] = self.trained.ticker.iloc[0]     # 增加ticker字段，注：这种写法仅对一个episode一个股票有效
        df['asset_pct_chg'] = self.asset_change_eps()  # 增加asset_pct_chg字段
        self.ds.append_train_history(df) # train_history，predicted和action_history是append的
        
        return df
