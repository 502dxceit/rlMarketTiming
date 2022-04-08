import sys,os
sys.path.append(os.getcwd())
import streamlit as st
from streamlit_option_menu import option_menu
import cufflinks as cf
import pandas as pd
import numpy as np
from datetime import datetime
from data_work import DataStorage
from predict import MODEL_PATH, WatchList, Predictor
import sqlite3

def load_css(file_name:str = "streamlit.css")->None:
    """
    Function to load and render a local stylesheet
    """
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

class WebServer:
    '''
        从上到下展示图表：
        · WatchList的最新预测，表格 [date,ticker,name,action in (buy,hold,sell)]， 取最后一日的结果 wl[-1]
        · WatchList的最新预测，半年股价图：x: date, y = close price， 叠加landmark和action 

        · 上一轮（episode/股票）的训练结果：x : date, y: close price 叠加landmark和action。
        · 上一轮（episode/股票）的训练结果：x : date, y: reward
        · 上一轮（episode/股票）的训练结果：x : date, y: asset增长比例
        · 上述3个图取最近半年详细显示
        
        · 历史以来的reward增长：x = epsisode，y : reward
        · 历史以来的asset增长：x = epsisode，y : asset_percent_change
        · 上述2个图取最近半年详细显示
        · 历史以来的操作记录df.columns = ['date','ticker','landmark','close','action','reward','asset_pct_change']
          >  这里的asset_pct_change是相比于上次动作的资产变化，asset_pct_change = df.close.diff()/close，如果为下跌并且action正确得换正号
          >  总资产的变化应当动态计算 total_change = reduce(lambda a,b:1+b, d.asset_pct_change)
    '''
    def load(self,from_table:str) -> pd.DataFrame:
        return pd.read_sql('SELECT * FROM %s'% from_table, con = self.conn,index_col='index')

    def save(self, df:pd.DataFrame, to_table:str, if_exists='replace'):
        return df.to_sql(name=to_table,con = self.conn, if_exists = if_exists)

    def __init__(self) -> None:
        self.conn = sqlite3.connect('stock.db')
        self.ds = DataStorage()
        self.predicted = self.ds.load_predicted() # watchlist_actions columns = ['ticker','date','close','action']
        self.watchlist_trend = Predictor(MODEL_PATH, WatchList).watchlist_trend()  # watchlist_trend.columns = ['ticker','date','close','landmark']

        # evaluated.columns = ['date','ticker','landmark','close','action','reward','asset_pct_chg'] 
        self.evaluated = self.ds.load_evaluated() # for price/reward/asset plotting
        # action_history结构和evaluated一样，但去掉了action==0的记录，供长期保存
        self.action_history = self.ds.load_action_history() 
        # train_history.columns =  ['episode','ticker','train_date','mean','std','asset_change']
        self.train_history = self.ds.load_train_history()

        # self.df = pd.DataFrame(np.random.randn(1000, 6), columns=['date','ticker','landmark','close','action','reward']).cumsum()
        # length = len(self.df)
        # self.df[['date']] = datetime.now().strftime("%Y-%m-%d")
        # self.df[['ticker']] = 'SZ.600283'
        # self.df[['action']] = np.random.choice([-1,0,1], (length,1)) 

        # self.df2 = pd.DataFrame(np.random.randn(100, 4), columns=['date','ticker','close','action']).cumsum()
        # length2 = len(self.df2)
        # self.df2[['date']] = datetime.now().strftime("%Y-%m-%d")
        # self.df2[['ticker']] = 'SZ.600283'
        # self.df2[['name']] = np.random.rand(length2,1) * 2 -1
        # self.df2[['action']] = np.random.choice([-1,0,1], (length2,1)) 
    

    def run(self):
        # st.set_page_config(
        #     page_title="Ex-stream-ly Cool App",
        #     page_icon="🧊",
        #     layout="wide",
        #     initial_sidebar_state="expanded",
        #     menu_items={
        #         'Get Help': 'https://www.extremelycoolapp.com/help',
        #         'Report a bug': "https://www.extremelycoolapp.com/bug",
        #         'About': "# This is a header. This is an *extremely* cool app!"
        #     }
        # )
        selected = option_menu("RL Market Timing", ["WatchList", 'Latest',"History"], 
                icons=['house', 'gear','palette'], menu_icon="cast", default_index=1,  orientation="horizontal")
        st.title("RL Market Timing")
        st.metric(label="Current Jobs", value="retrieving SH.000234", delta="323 records")
        col1, col2 = st.columns(2)
        col1.metric(label="Reward", value="0.902342", delta="+0.3")
        col2.metric(label="Assets", value="$7,000,000", delta="+1200")
        st.header("WatchList - ACTIONS")
        df = self.df2[['date','ticker','action']]
        df[['action']] = df[['action']].applymap(lambda x:{-1:"sell",0:"hold",1:"buy"}[x])
        st.table(df.sample(4))  # watchlist table, .write(), .table(), .dataframe()
        # 下面两行都可以，绘图软件尝试换成plotnine
        # fig = df.iplot(asFigure=True,subplots=True,shape=(3,2),mode='lines+markers',theme='ggplot')
        fig = self.df[-100:][['close','action','reward']].figure(subplots=True,shape=(3,1),mode='lines+markers',theme='ggplot') 
        st.plotly_chart(fig)
        st.header("WatchList - Last 6 months")
        st.line_chart(self.df[-50:]['close'])
        st.header("Latest Training - PRICE & ACTIONS")
        st.line_chart(self.df['close'])
        st.header("Latest Training - REWARDS")
        st.line_chart(self.df['close'])
        st.header("Latest Training - ASSET GROWTH")
        st.line_chart(self.df['close'])
        st.header("Latest Training - PRICE & ACTIONS in last 6 months")
        st.line_chart(self.df[-50:]['close'])
        st.header("Latest Training - REWARDS in last 6 months ")
        st.line_chart(self.df[-50:]['close'])
        st.header("Latest Training - ASSET GROWTH in last 6 months")
        st.line_chart(self.df[-50:]['close'])
        st.header("History - REWARDS")
        st.line_chart(self.df['close'])
        st.header("History - ASSET GROWTH")
        st.line_chart(self.df['close'])
        st.header("History - REWARDS in last 6 months")
        st.line_chart(self.df['close'])
        st.header("History - ASSET GROWTH in last 6 months")
        st.line_chart(self.df['close'])
        st.header("History - Actions Table")
        st.dataframe(df)


if __name__=='__main__':
    app = WebServer()
    load_css()
    app.run()