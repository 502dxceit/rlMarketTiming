from re import A
import sys,os
import time, datetime
sys.path.append('./DQN')
sys.path.append('./vae')
sys.path.append('.')


import schedule
import torch
from retrying import retry

from data_work import DataWorker, DataStorage
from preprocess import Preprocessor
from predict import Predictor, WatchList
from evaluate import Evaluator

from visualize2 import WebServer
from utils import time_cost
from globals import MAIN_PATH
import pysnooper
from model import MLP
from env import StockMarketEnv
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from collector import stock_Collector

import tianshou as ts


def test_webserver():
    ws = schedule.get_jobs("webserver")
    if ws is None:
        print("web server down, restarting")
        WebServer().run()

def train_job():
    env = StockMarketEnv()
    train_envs = ts.env.DummyVectorEnv([lambda: StockMarketEnv() for _ in range(8)])
    # test_envs = ts.env.DummyVectorEnv([lambda: StockMarketEnv() for _ in range(4)])
    state_shape = env.state_space
    action_shape = env.action_space.n
    net = MLP(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)

    train_collector = stock_Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 8), exploration_noise=True)
    test_collector = None # ts.data.Collector(policy, test_envs)

    writer = SummaryWriter()
    writer.add_text(text_string="test", tag="test_arg")
    logger = TensorboardLogger(writer)

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10, step_per_epoch=10000, step_per_collect=8,
        update_per_step=0.1, episode_per_test=100, batch_size=64,           # update per step 为何之前是0.1，在cartpole上能训练起来？
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),               # 1. update是用for _ in range(round(update_per_step* result[n/st]))
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),               # 为何之前的step_per_collect 写成了2？ 猜测是测试并行env为2 时误写
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
        logger=logger)

    print(result)

if __name__ == "__main__":
    train_job()

