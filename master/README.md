# rlMarketTiming

强化学习股市择时策略实验项目

该项目是一个基于强化学习（Reinforcement Learning）的股市预测问题，程序的最终目标是给出择时策略。


在这个项目中使用的强化学习：

     ┌--->----state, reward---->------┑
     |                                |
    env(env.py: StockMarketEnv)      agent(tianshou.policy.DQNpolicy)
     |                                |
     ┕---<----action------<-----------┙

主程序在main.py，分为三个模块，train, predict, webserver

注：整个股票数据存储都通过data_work.py: DataStorge读存

train流程:
    获取随机股票日线(data_work.py: DataWorker)，进行预处理(preprocess.py: Preprocessor)后存入当前环境StockMarketEnv中，之后进行强化学习，过程中会反复调用上述的步骤
    。。。
    preprocess后的股票数据暂存在stock.db 的downloaded表里
    train的reward数据暂存在rew.db 的res表里

predict流程：
    对watchlist里的股票进行预测

webserver:
    将predict和train的结果，以及一些其他部分展示在对外网开放服务器应用上。

