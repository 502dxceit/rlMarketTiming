#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 21:14:12
LastEditor: John
LastEditTime: 2021-05-04 02:45:27
Discription: 
Environment: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

import pickle
# class MLP(nn.Module):
#     def __init__(self, input_dim,output_dim,hidden_dim=128):
#         """ 初始化q网络，为全连接网络
#             input_dim: 输入的feature即环境的state数目
#             output_dim: 输出的action总个数
#         """
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim) # 输入层
#         # self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim) # 隐藏层
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)
#         # self.bn2 = nn.BatchNorm1d(hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, output_dim) # 输出层
        
#     def forward(self, x):
#         # 各层对应的激活函数
#         # if x.shape.__len__() == 1:
#             x = F.relu(self.fc1(x))
#         # x = self.bn1(x)
#             x = F.relu(self.fc2(x))
#         # x = self.bn2(x)
#             x = F.relu(self.fc3(x))
#             return self.fc4(x)
#         # else:
#         #     x = F.relu(self.fc1(x))
#         #     x = self.bn1(x)
#         #     x = F.relu(self.fc2(x))
#         #     x = self.bn2(x)
#         #     return self.fc3(x)
# 弃用，用适配tianshou 的network写法
            
class MLP(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )
        self.output_dim = action_shape

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))

        # with open("output.pkl", "wb") as file:
        #     pickle.dump((obs, logits), file)

        # print(obs, logits)
        # breakpoint()
        return logits, state