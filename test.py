# import sqlite3
# import pandas as pd
# conn = sqlite3.connect('stock.db')
# df = pd.read_sql("SELECT * FROM downlowded", conn)

# print(df)

# # # from globals import MAIN_PATH
# # # print(MAIN_PATH)
# # print(df)
# import data_work
# from preprocess import Preprocessor
# import torch
# import numpy as np
# from torch import Tensor

# tensor = Tensor([ 0.5656,  0.1255, -0.8877])
# print(tensor.tolist().index(tensor.max()))


# # ds = data_work.DataStorage(r"E:\rlMarketTiming", "stock.db")

# conn = sqlite3.connect(r"E:\rlMarketTiming\stock.db")
# print(pd.read_sql("SELECT * FROM downloaded", conn))
# print([0]*10)
# dict = {"1": 1, "2": 2}
# import pandas as pd
# print(pd.Series(pd.Series(dict)))

# dw = data_work.DataWorker().get()
# ds.save_raw(dw)

# ppcs_df = Preprocessor().bundle_process()

# # print(ppcs_df)

# ds.save_processed(ppcs_df)

# print(ds.load_processed().columns)

# from ast import Return
# import pysnooper

# def test():
#     A = 1
#     b = 2
#     return A - b

# @pysnooper.snoop()
# def func():
#     return test()

# func()

# pysnooper 只会Log最外层

import pandas as pd
import numpy as np
df = pd.DataFrame(np.zeros((3,3)))
print(df.loc[df.index[-1], 0])

# class A:
#     def __init__(self):
#         self.p = 1

# A().start()

# import sqlite3
# import pandas as pd

# df = pd.read_sql("SELECT * FROM downlowded_tarined", sqlite3.connect('stock.db'))
# print(df.columns)

