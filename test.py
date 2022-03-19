import sqlite3
import pandas as pd
# conn = sqlite3.connect('DQN/stock.db')
# df = pd.read_sql("SELECT * FROM downlowded", conn)
# # from globals import MAIN_PATH
# # print(MAIN_PATH)
# print(df)
import data_work
from preprocess import Preprocessor
import torch
import numpy as np
from torch import Tensor

tensor = Tensor([ 0.5656,  0.1255, -0.8877])
print(tensor.tolist().index(tensor.max()))


# ds = data_work.DataStorage(r"E:\rlMarketTiming", "stock.db")

# # conn = sqlite3.connect(r"E:\rlMarketTiming\stock.db")
# # print(pd.read_sql("SELECT * FROM downloaded", conn))



# dw = data_work.DataWorker().get()
# ds.save_raw(dw)

# ppcs_df = Preprocessor().bundle_process()

# # print(ppcs_df)

# ds.save_processed(ppcs_df)

# print(ds.load_processed().columns)
