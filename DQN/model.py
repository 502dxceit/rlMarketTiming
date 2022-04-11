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


class MLP(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim=128):
        """ 初始化q网络，为全连接网络
            input_dim: 输入的feature即环境的state数目
            output_dim: 输出的action总个数
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) # 输入层
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # 隐藏层
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim) # 输出层
        
    def forward(self, x):
        # 各层对应的激活函数
        if x.shape.__len__() == 1:
            x = F.relu(self.fc1(x))
        # x = self.bn1(x)
            x = F.relu(self.fc2(x))
        # x = self.bn2(x)
            return self.fc3(x)
        else:
            x = F.relu(self.fc1(x))
            x = self.bn1(x)
            x = F.relu(self.fc2(x))
            x = self.bn2(x)
            return self.fc3(x)
            
