from DQN.agent import DQN
import torch
import gym
from torch import nn
import numpy as np

config = {
    "algo": "DQN",
    "train_eps": 1000, # 100
    "eval_eps": 5,
    "gamma": 0.9,
    "epsilon_start": 0.90,
    "epsilon_end": 0.01,
    "epsilon_decay": 500,
    "lr": 1e-3,
    "memory_capacity": 100000,
    "batch_size": 64,
    "target_update": 320,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "hidden_dim": 128, 
    "save_model": 1,   # 多少episode保存一次模型
    "save_result": 1,   # 多少episode保存一次结果
    "change_tic": 20
}   # copy from train.py

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        logits = self.model(obs)
        return logits


agent = DQN(4, 2, **config)
net = Net(4, 2)
# net = torch.load("nice.pth")
# agent.policy_net = net
env = gym.make("CartPole-v1")
total_step = 0
for i in range(config["train_eps"]):
    state = env.reset()
    rewards = 0
    while True:
        total_step += 1
        action = agent.choose_action(state)
        state_, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, state_, done)
        agent.update()
        rewards += reward
        state = state_
        if done:
            break

    print("train episode{}".format(str(i)), "total_reward:{}".format(str(rewards)), "total_step{}".format(total_step))

