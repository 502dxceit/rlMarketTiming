# CAUTION: globals variables could be imported by other modules widely, so don't do it wildly. 
# Any modification may lead to unpredictable consequence!
import os
import baostock

baostock.login(user_id="anonymous", password="123456")

import os
MAIN_PATH = os.path.dirname(__file__)  # 当前文件所在的目录
os.chdir(MAIN_PATH)
print(f"Working in {MAIN_PATH}")

# indicators = ['kdjk', 'kdjd', 'kdjj', "rsi_6", "rsi_12", "rsi_24",'cr',"boll","boll_ub","boll_lb","wr_10","wr_6","cci","dma"] # 14
indicators = ['kdjk', 'kdjd', 'kdjj', "rsi_6", "rsi_12", "rsi_24","macd","atr"]
oclhva = ["open", "close", "high", "low", "volume","amount"]  # 6 fields need to be normalized
oclhva_after = ["open_", "close_", "high_", "low_", "volume_","amount_"] # 6 fields after normlization

mkt_indicators = indicators
mkt_indicators_after = ['mkt_kdjk', 'mkt_kdjd', 'mkt_kdjj', "mkt_rsi_6", "mkt_rsi_12", "mkt_rsi_24",\
    'mkt_cr',"mkt_boll","mkt_boll_ub","mkt_boll_lb","mkt_wr_10","mkt_wr_6","mkt_cci","mkt_dma"]
mkt_oclhva = oclhva
mkt_oclhva_after = ["mkt_open_", "mkt_close_", "mkt_high_", "mkt_low_", "mkt_volume_","mkt_amount_"] # fields after normlization

window_size = 20

from collections import namedtuple
ACTION = {'buy':1, 'hold':0, 'sell':-1 }
# 字典这么操作，就可以如 actions.buy 这样写
actions = namedtuple("Action",ACTION)(**ACTION)     