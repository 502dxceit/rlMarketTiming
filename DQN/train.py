#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:48:57
@LastEditor: Yin Tang
@LastEditTime: 2022-2-6
@Discription: 
@Environment: python 3.8
'''

import sys,os
sys.path.append(os.getcwd())
sys.path.append('..') # 需要引用父目录的data_work, preprocess, utils
# print(sys.path)
from collections import namedtuple
import torch
import pysnooper
from retrying import retry
from DQN.agent import DQN
from DQN.env import StockMarketEnv
from data_work import DataStorage
from preprocess import Preprocessor
from utils import time_cost, error_report,createdir_if_none

config = {
    "algo": "DQN",
    "train_eps": 100,
    "eval_eps": 5,
    "gamma": 0.95,
    "epsilon_start": 0.90,
    "epsilon_end": 0.01,
    "epsilon_decay": 500,
    "lr": 0.0001,
    "memory_capacity": 100000,
    "batch_size": 64,
    "target_update": 4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "hidden_dim": 256, 
    "save_model": 1,   # 多少episode保存一次模型
    "save_result": 1,   # 多少episode保存一次结果
    "change_tic": 20
}

# convert dict to namedtuple for more pythonic access
# visit its property as 'cfg.batch_size'
cfg = namedtuple("Config",config)(**config) 

class Trainer:
    def __init__(self, config, agent, env) -> None:
        self.config = config
        self.agent = agent
        self.env = env
        self.result_dir = './result/'
        self.model_dir = './model/'
        self.image_dir = './image/'
        self.ds = DataStorage()
        self.env.make()

    def create_data_dir(self):
        createdir_if_none(self.model_dir)
        # createdir_if_none(self.result_dir)
        # createdir_if_none(self.image_dir)
    
    def load_data(self):
        return self.ds.load_processed()

    def save_data(self,actions:list[int], rewards:list[float]):
        '''actions is a list of actions, for performance evaluation'''
        self.df['action'] = actions
        self.df['reward'] = rewards
        return ds.save_trained(self.df)

    def load_model(self):
        self.agent.load()

    def save_model(self, episode):
        '''save the model to designated path'''
        self.agent.save()
        # torch.save(self.agent.target_net.state_dict(), self.model_dir + str(episode) +".pth")

    @time_cost
    def train(self):
        print(f'Env:{self.env}, Algorithm:{self.config["algo"]}, Device:{self.config["device"]}')
        for episode in range(self.config["train_eps"]): # 若一个episode是一个股票，那么我们似乎只需要一层循环即可
            state = self.env.reset()
            running_reward = 0

            @retry(retry_on_exception=error_report,stop_max_attempt_number=5) # a more elegant error_handler than "try-except"
            def temfun():
                actions,rewards =[], []
                done = False
                step = 0
                while step < (len(self.env.df) - 2*self.env.window_size):
                    step += 1
                    action = self.agent.choose_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.agent.memory.push(state, action, reward, next_state, done)
                    state = next_state
                    rewards.append(reward)
                    actions.append(action)
                    running_reward = running_reward*0.9 + reward *0.1
                    self.agent.update()  # backpropagates to update the NN
                    if done:
                        break

                return actions, rewards
            actions, rewards = temfun()
            # if (episode + 1) % self.config["save_model"] == 0:
            self.agent.save(episode)  # save DQN model to pth
            self.ds.save_trained(actions,rewards) # save trained data to table 'trained', appending 'actions', 'rewards'

            if (episode + 1) % self.config["target_update"] == 0:  # update the target_net from policy_net
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
            # if (episode + 1) % self.config['change_tic'] == 0:
            #     print("change at {}".format(episode))
            #     time.sleep(30)
            #     self.env.creat_df()
            
        return actions, rewards

if __name__ == "__main__":
    '''load processed data, initiate training, then save to trained'''
    ds = DataStorage()
    df = ds.load_processed()
    env = StockMarketEnv(df)  # StockLearningEnv() requires a series of embeddings as input 
    agent = DQN(env.state_space, env.action_space.n, **config)
    trainer = Trainer(config, agent, env)
    trainer.create_data_dir() 
    actions, rewards = trainer.train()
    trainer.save_data(actions, rewards)