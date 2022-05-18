# import sqlite3
# import pandas as pd
# conn = sqlite3.connect("stock5.db")

# df = pd.read_sql("select * from downloaded", conn)
# from typing import List

# import gym

# # from preprocess import Preprocessor

# # pre = Preprocessor()



# # import matplotlib.pyplot
# import matplotlib.pyplot as plt
# import pandas as pd\
# # # y3 = env.df.iloc[-2], env.df.iloc[-1]

# # # print(y3)

# import matplotlib.pyplot as plt
import numpy as np

# print(env.df.columns)
# for j in range(10):
#     env.reset()
#     for i in env.df.iloc[:, 2:].columns:
#         if i != "landmark":
#             L2 = np.linalg.norm(np.isinf(env.df.loc[:, i].values))
#             if L2 !=0:
#                 print(i)

# print()
# for i in range(10):
#     env.reset()
#     rw = 0
#     len = 0
#     while True:
#         s, r, d, _ = env.step(env.action_space.sample())
#         rw += r
#         len += 1
#         if d:
#             break
#     print(rw, len)

import pandas as pd
import numpy
# print(pd.DataFrame(numpy.array([1, 2, 3, 4, 5])).T)
import sqlite3
# import matplotlib.pyplot as plt
from seaborn import lineplot
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.relplot

def move_average(x= 5000):
    df = pd.read_sql("select * from res", sqlite3.connect("rew0.db"))
    df.columns = ["step", "reward"]
    ma = df.loc[:, "reward"].rolling(x).mean().values
    df2 = pd.DataFrame(data=np.concatenate([np.arange(ma.shape[0]).reshape((ma.shape[0], 1)), ma.reshape((ma.shape[0], 1))], axis=1), columns=["step", "reward"])
    df2 = df2.dropna()
    sns.relplot(data=df, x="step", y="reward", ci="sd", kind="line")
    # plt.plot(df)
    # plt.show()
    # sns.lineplot(data=df, x="step", y="reward", ci=90, err_style="ci_band")
    # sns.relplot(data=df, x="step", y="reward", ci="sd", kind="line")
    plt.show()
    return df2



print(move_average())
# r_ = []
# while True:

#     s, r, d, _ = env.step(2)
#     r_.append(r)

#     if d:
#         env.step(2)
#         env.step(2)
#         env.step(2)
#         env.step(2)
#         s, r, d, _ = env.step(2)
#         print(s, r, d, _)
#         break

# # print(r_.__len__(), env.df.__len__())
# landmark = env.df.landmark[:200]
# print(landmark[landmark.isin(["^", "v"])])

# plt.subplot(211)
# plt.plot(env.df.close[:200])
# plt.subplot(212)
# plt.plot(r_[:200])
# # print(r_[:200], env.df.close[:200])
# plt.show()


# print(pd.read_csv("extremums.csv").values, type(pd.read_csv("extremums.csv").values))
# env = gym.make("CartPole-v1")
# print(env.spec.reward_threshold)

# import numpy as np
# from numpy import array
# rew_stack, done_stack, info_stack = map(np.stack, [(array([0.8875445]), 0), (False, False), ({'env_id': 0}, {'env_id': 1})])

# print(rew_stack, done_stack, info_stack)


# net = MLP(280, 3)

# arr = np.random.randn(63, 280)

# arr = np.concatenate([arr, np.array([[np.inf]*280])], axis=0)
# print(arr)
# print(net(torch.Tensor(arr)))

# import pickle 
# import torch

# with open("output.pkl", "rb") as file:
#     obs, q = pickle.load(file)

# tensor = torch.concat([obs, q], dim=1)

# print(tensor[1])

# import pickle 
# import torch

# with open("output2.pkl", "rb") as file:
#     data = pickle.load(file)


# # print(data.obs)
# for i in data.obs:
#     print(i)

