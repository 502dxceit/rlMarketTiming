import streamlit as st
from streamlit_option_menu import option_menu
import cufflinks as cf
import pandas as pd
import numpy as np
from datetime import datetime
from data_work import DataStorage
from predict import MODEL_PATH, WatchList, Predictor
import sqlite3

class WebServer:
    def load(self,from_table:str) -> pd.DataFrame:
        return pd.read_sql('SELECT * FROM %s'% from_table, con = self.conn,index_col='index')

    def __init__(self) -> None:
        self.conn = sqlite3.connect('stock.db')
        # watchlist_actions columns = ['ticker','date','close','action']
        # self.predicted = self.load("action_history")
        # watchlist_trend.columns = ['ticker','date','close','landmark']
        # self.watchlist_trend = self.load("predicted")
        # evaluated.columns = ['date','ticker','landmark','close','action','reward','asset_pct_chg'] 
        self.evaluated = self.load("downlowded_tarined")
        # action_history结构和evaluated一样，但去掉了action==0的记录，供长期保存
        # self.action_history = self.load("action_history")
        # train_history.columns =  ['episode','ticker','train_date','mean','std','asset_change']
        self.train_history = self.load("train_history")

    def run(self):
        selected = option_menu("RL Market Timing", ["WatchList", 'Latest',"History"], 
                icons=['house', 'gear','palette'], menu_icon="cast", default_index=1,  orientation="horizontal")
        st.title("RL Market Timing")
        st.metric(label="Current Jobs", value="retrieving SH.000234", delta="323 records")
        col1, col2 = st.columns(2)
        col1.metric(label="Reward", value="0.902342", delta="+0.3")
        col2.metric(label="Assets", value="$7,000,000", delta="+1200")
        st.header("WatchList - ACTIONS")
        df = self.evaluated[['date','ticker','action']]
        # df[['action']] = df[['action']].applymap(lambda x:{-1:"sell",0:"hold",1:"buy"}[x])
        st.table(df.sample(4))  # watchlist table, .write(), .table(), .dataframe()
        # 下面两行都可以，绘图软件尝试换成plotnine或seaborn
        # fig = df.iplot(asFigure=True,subplots=True,shape=(3,2),mode='lines+markers',theme='ggplot')
        fig = self.evaluated[-100:][['close','action','reward']].figure(subplots=True,shape=(3,1),mode='lines+markers',theme='ggplot') 
        st.plotly_chart(fig)
        st.header("WatchList - Last 6 months")
        st.line_chart(self.evaluated[-50:]['close'])
        st.header("Latest Training - PRICE & ACTIONS")
        st.line_chart(self.evaluated['close'])
        st.header("Latest Training - REWARDS")
        st.line_chart(self.evaluated['close'])
        st.header("Latest Training - ASSET GROWTH")
        st.line_chart(self.evaluated['close'])
        st.header("Latest Training - PRICE & ACTIONS in last 6 months")
        st.line_chart(self.evaluated[-50:]['close'])
        st.header("Latest Training - REWARDS in last 6 months ")
        st.line_chart(self.evaluated[-50:]['close'])
        st.header("Latest Training - ASSET GROWTH in last 6 months")
        st.line_chart(self.evaluated[-50:]['close'])
        st.header("History - REWARDS")
        st.line_chart(self.evaluated['close'])
        st.header("History - ASSET GROWTH")
        st.line_chart(self.evaluated['close'])
        st.header("History - REWARDS in last 6 months")
        st.line_chart(self.evaluated['close'])
        st.header("History - ASSET GROWTH in last 6 months")
        st.line_chart(self.evaluated['close'])
        st.header("History - Actions Table")
        st.dataframe(df)


if __name__=='__main__':
    app = WebServer()
    app.run()