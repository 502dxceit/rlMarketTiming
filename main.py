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
from DQN.env import StockMarketEnv

from DQN.agent import DQN
from DQN.train import Trainer, config
from visualize import WebServer
from utils import time_cost
from globals import MAIN_PATH

def test_webserver():
    ws = schedule.get_jobs("webserver")
    if ws is None:
        print("web server down, restarting")
        WebServer().run()

def train(ts_embedded):
    env = StockMarketEnv(ts_embedded)
    agent = DQN(env.state_space, env.action_space.n, **config)
    trainer = Trainer(config, agent, env)
    trainer.create_data_dir()
    res= trainer.train()    # res: tuple(actions[], rewards[])
    
    return res[1]

@time_cost
def train_job():
    ds = DataStorage()
    raw_data = DataWorker().get() # 1 get data from tushare     # DataWorker.get() -> DataWorker().get()
    ds.save_raw(raw_data)
    train_data = Preprocessor(df=raw_data).bundle_process() # 2 do preprocess
    ds.save_processed(train_data) 
    # train_data['embedding'].apply(lambda x:float(x)) # embedding 不是形如 "AAAAA, BBBBB, CCCCC"的向量吗？
    # zz = [torch.Tensor(list(map(float, v.split(",")))) for v in train_data.embedding] # a series of embeddings
    # zz = strSeries2TensorList(data.embedding)
    reward = train(train_data)  # []
    train_data["reward"] = reward + [0]*(train_data.__len__()-reward.__len__()) # train(zz) # 3 train on data with model saved in model.pth and signals in table 'evaluated'
    # 仍然存在的问题，输出的reward序列为何不与原本的train序列等长？

    
    # ds.save_trained(train_data[['ticker','date','close','action','reward']])  # train_save_data里已经save了
    # evaluated = Evaluator(train_data).start() # 4 evaluates the results -- whether or not, or how much signals match the landmarks
    # ds.save_evaluated(evaluated)  #这两行放到train里，每个episode后都evaluated并保存


@time_cost
def predition_job():
    predictor = Predictor() 
    actions = predictor.watch()
    return actions    

if __name__ == "__main__":
    import pysnooper
    
    MAIN_PATH = os.getcwd()
    print("working on ",MAIN_PATH)
    # WebServer().run()
    # @pysnooper.snoop("outlog.txt")
    # pysnooper at evaluate.save_train_history
    # def func():
    #     train_job()
    # func()    # debugged fine
    # predition_job()   # debugged fine
    '''

    schedule.every(12).hours.at("00:00").do(train_job).tag('hour-tasks', 'hour') # the training process
    schedule.every().day.do(predition_job).tag("daily-tasks","day") # prediction stands alone
    schedule.every(30).minutes.do(test_webserver).tag("webserver","30m") # webserver/visualizer listens 7/24, occasionally tested

    while True:
        schedule.run_pending()
        all_jobs = schedule.get_jobs()
        print("### JOBS List ###")
        print(all_jobs)
        time.sleep(2)
    '''

