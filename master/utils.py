import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import torch
import numpy as np
import os
import pysnooper
from retrying import retry
import numpy as np
import matplotlib.ticker as ticker

## MDPPplus有bug，逐步废弃
def MDPPplus(df:pd.DataFrame, D = 10,P = 0.05)->pd.DataFrame:
    '''
        finding the peak ^,bottom v of a series with D days and P percentage of fluctuation
        Landmarks: a new model for similarity-based pattern querying in time series databases
        https://ieeexplore.ieee.org/document/839385
    '''
    df['landmark'] = '-'
    df.loc[(df.close.diff(1) > 0) & (df.close.diff(-1) > 0) & (df.landmark != 1), 'landmark'] =  '^'
    df.loc[(df.close.diff(1) < 0) & (df.close.diff(-1) < 0) & (df.landmark != 1), 'landmark'] =  'v'
    d = df[df['landmark'].isin(['^', 'v'])]
    for i in range(0,len(d)-2,2):
        if within_mdpp(d.index[i],d.loc[d.index[i],'close'],d.index[i+1],d.loc[d.index[i+1],'close']):
            df.loc[d.index[i]: d.index[i+1],'landmark'] = '-'
    return df
###################
def within_mdpp(x1,y1,x2,y2,D,P) -> bool:
    ''' whether or not (x1,y1) and (x2,y2) meets the MDPP criteria, can also be defined as:
        within_mdpp  = lambda x1,y1,x2,y2,D,P:(abs(x2-x1)<D) and (abs(y2-y1)*2/abs(y1+y2)<P) 
    '''
    return (abs(x2-x1)<D) and (abs(y2-y1)*2/abs(y1+y2)<P)

def base_landmarks(df:pd.DataFrame):
    '''把所有拐点全拿出来,并在df上做标记增加字段landmark'''
    d = df.copy()
    d.close = d.close.astype("float")
    # 先找出所有拐点，其他设为 “-”
    d['landmark'] = '-'
    # y-x>0 and y-z < 0 /   -     d.diff(1)>0 and d.diff(-1)<0
    # y-x<0 and y-z > 0 \   -     d.diff(1)<0 and d.diff(-1)>0
    # y-x>0 and y-z > 0 ^   peak  d.diff(1)>0 and d.diff(-1)>0
    # y-x<0 and y-z < 0 v   bottom   d.diff(1)<0 and d.diff(-1)<0
    d.loc[(d.close.diff(1)>0)&(d.close.diff(-1)<0),'landmark'] = '/'  # 上涨中
    d.loc[(d.close.diff(1)<0)&(d.close.diff(-1)>0),'landmark'] = '\\' # 下跌中
    d.loc[(d.close.diff(1)>0)&(d.close.diff(-1)>0),'landmark'] = '^'  # 波峰
    d.loc[(d.close.diff(1)<0)&(d.close.diff(-1)<0),'landmark'] = 'v'  # 波谷
    return d[d.landmark.isin(['^','v'])]  # .index 找出所有拐点

    
def landmarks(df:pd.DataFrame, D = 10, P=0.05):
    '''计算data的界标，
    [1] Perng C S, Wang H, Zhang S R, et al. Landmarks: a new model for similarity-based pattern querying in time series databases[C]// International Conference on Data Engineering, 2000. Proceedings. IEEE, 2000:33-42.
    href="http://citeseer.ist.psu.edu/viewdoc/download?doi=10.1.1.120.3361&rep=rep1&type=pdf"
    remove two adjacent points $(x_i,y_i) ,(x_{i+1},y_{i+1})$ if  $x_{i+1}-xi < D$ and $\frac{|y_{i+1}-y_i|}{(|y_i|+|y_{i+1|)/2}} <P$
    '''
    bl = base_landmarks(df)     # 找出所有的拐点
    # print(df, bl) #某些df行数太少导致landmark不出点报错
    index_list = bl.index.to_list()
    # v1  优雅了优雅了，虽然有bug，不能用  
    # for _ in range(D):
    #     d = d[~(2*abs(d.close.diff(1))/(abs(d.close)+abs(d.close.shift(1)))<P)] # 找出相邻点满足变化<P的数据行
    # 先来一个丑陋的版本顶着用 v2
    # lm = [
    #     index_list[i]
    #     for i in range(len(index_list)-1)
    #     if not within_mdpp(index_list[i],bl.loc[index_list[i]].close,index_list[i+1],bl.loc[index_list[i+1]].close,D,P)
    # ] 
    ## 先拿个丑陋的版本顶着用 v3
    lm = []
    p1,p2 = 0,1
    
    lm.append(index_list[p1])
    while p1< len(index_list):
        if within_mdpp(index_list[p1],bl.loc[index_list[p1]].close,index_list[p2],bl.loc[index_list[p2]].close,D,P):
            p2 = p2 + 1
            if p2 >= len(index_list): break
        else:
            p1 = p2            
            lm.append(index_list[p1])
    #############################
    d = bl.loc[lm] #经过mdpp过滤后的坐标集合
    d = base_landmarks(d) # 注意加上这行的重要不同，将重新洗牌，实现peak和bottom交替
    return d[d.landmark == '^'].index, d[d.landmark == 'v'].index

def landmarks_BB(df:pd.DataFrame):
    d=df[['close']]
    dataShifted = pd.DataFrame(index = d.index)
    for i in range(-5, 5):
        dataShifted = pd.concat([dataShifted, d.shift(i).rename(columns = {d.columns[0]: 'shift_' + str(i)})], axis = 1)
    dataInd = pd.DataFrame(0, index = d.index, columns = d.columns)
    dataInd[dataShifted['shift_0'] >= dataShifted.drop('shift_0', axis = 1).max(axis = 1)] = 1
    dataInd[dataShifted['shift_0'] <= dataShifted.drop('shift_0', axis = 1).min(axis = 1)] = -1
    dataInd[:5] = 0
    dataInd[-5:] = 0

    extremums=dataInd[dataInd.isin([-1,1])].dropna().iloc[:,0] # 波峰波谷的索引序列
    # print(extremums)
    duplicate_index_real=duplicate_index(extremums) # 重复波峰\波谷的绝对坐标
    # print(duplicate_index_real)
    x_list=[]
    for dup_list in duplicate_index_real:
        ld=extremums.loc[dup_list[0]]
        x = df[['close']].loc[dup_list,:].idxmax().iloc[0] if ld==1 else df[['close']].loc[dup_list,:].idxmin().iloc[0]
        x_list.append(x)
    # print(x_list)
    it=iter(x_list)
    for lst in duplicate_index_real:
        lst.remove(next(it))
        extremums.drop(index=lst,inplace=True)

    return extremums[extremums==1].index,extremums[extremums==-1].index

def attach_to(df:pd.DataFrame,peaks:pd.Index,bottoms:pd.Index)->pd.DataFrame:
    df['landmark'] = '-'
    df.loc[peaks,'landmark'] = '^'
    df.loc[bottoms,'landmark'] = 'v'
    return df

def depict(s:pd.Series,peaks:pd.Index,bottoms:pd.Index)->None:
    plt.figure(figsize=(6,4),dpi=160)
    sns.lineplot(data=s)
    for p in peaks: plt.text(x=p, y=s.loc[p], s='x', color="#f03752") # 海棠红 http://zhongguose.com/#haitanghong
    for b in bottoms: plt.text(x=b, y=s.loc[b], s='o',color="#41ae3c") # 宝石绿 http://zhongguose.com/#baoshilv
    plt.show()

##########################

def duplicate_index(series,duplicate_num=2):
    """
    :param series: 目标dataframe某个序列值
    :param duplicate_num: 判断相邻重复值超过设定值就输出重复的索引
    :return: 每个计算周期内需要删除的绝对索引列表
    """
    series0 = series.reset_index(drop=True)
    slide_list = [series0.index[0]]
    slide_list_all = []
    for i in range(series0.index[0], series0.index[-1]):
        j = i + 1
        diff = series0[j] - series0[i]
        if diff == 0:
            slide_list.append(j)
        else:
            slide_list.clear()
            slide_list.append(j)
        # print("slide_list:",slide_list)
        if len(slide_list) >= duplicate_num:
            target_list = slide_list.copy()
            slide_list_all.append(target_list)
    # print("slide_list_all:",slide_list_all)
    index = []  # 将找到的满足条件的index合并
    # 因为可能有前后包含的情况，只保留最长序列
    for i in range(len(slide_list_all) - 1):
        if set(slide_list_all[i]) < set(slide_list_all[i + 1]):
            index.append(i)
    m = {i: element for i, element in enumerate(slide_list_all)}
    [m.pop(i) for i in index]
    # 将所有需要删除的行数合并
    # indexs_to_delete = []
    # for i in range(len(slide_list_all)):
    #     indexs_to_delete = list(set(indexs_to_delete).union(slide_list_all[i]))
    index = list(m.values())
    index_real = []  # 相邻重复序列的绝对坐标
    for i in index:  # 用相对坐标求绝对坐标
        lst = []
        for j in i:
            lst.append(series.index[j])
        index_real.append(lst)
    return index_real

def time_cost(func):
    # train 要求返回reward与action
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        res = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        print("{} completed,time used:{}".format(func.__name__,end_time - start_time))
        return res
    return wrapper

def inf2max(df:pd.DataFrame)->pd.DataFrame:
    '''replace np.inf with the max value in the column'''
    return 

def pdSeries2TensorList(embeddings:pd.Series):
    ''' embedded vector stored in df field is something like String "1,3,6,2.6,7,2,8".
        we have to convert them into a list of torch.Tensor:
        [tensor([1., 6., 2., 8.]), tensor([3., 2., 3., 7.]), tensor([1., 4., 1., 0.]), ... ... ]
    '''
    return [torch.Tensor(list(map(float, v.split(",")))) for v in embeddings]

def str2nparray(embedding:str):
    '''
        一般的DRL环境都是返回np.array，然后在训练的时候（自动）转换成tensor，
        当然这里也可以直接改用torch.Tensor()
    '''
    return np.array(list(map(float, embedding.split(","))),dtype=np.float64)

def error_report(exception):
    print("Err:",exception)
    return False    # 必须有返回项，否则File "e:\rlMarketTiming\DQN\train.py", line 109, in train
                    #     actions, rewards = temfun()
                    #   File "E:\Python3910\lib\site-packages\retrying.py", line 49, in wrapped_f
                    #     return Retrying(*dargs, **dkw).call(f, *args, **kw)
                    #   File "E:\Python3910\lib\site-packages\retrying.py", line 205, in call
                    #     if not self.should_reject(attempt):
                    #   File "E:\Python3910\lib\site-packages\retrying.py", line 189, in should_reject
                    #     reject |= self._retry_on_exception(attempt.value[1])
                    # TypeError: unsupported operand type(s) for |=: 'bool' and 'NoneType'

def createdir_if_none(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("{} 文件夹创建成功!".format(dir))
    else:
        print("{} 文件夹已存在!".format(dir))

    return dir

def plot_signal(df):
    # by: gwt
    # change log: 2020.10.27 for reference only
    image_dir = './outputs/image/'
    df["actions_signal"] = '-'
    df.loc[df[df.actions > -1].index, 'actions_signal'] = '^'
    df.loc[df[df.actions < -1].index, 'actions_signal'] = 'v'

    df_actions_sign = np.sign(df["actions"])
    buying_signal = df_actions_sign.apply(lambda x: True if x > -1 else False)
    selling_signal = df_actions_sign.apply(lambda x: True if x < -1 else False)

    tic_plot = df['close']
    tic_plot.index = df.index

    plt.figure(figsize=(19, 12))
    plt.plot(tic_plot, color='steelblue', lw=1.)
    plt.plot(tic_plot, '^', markersize=2, color='r', label='buying signal', markevery=buying_signal,alpha=0.3)
    plt.plot(tic_plot, 'v', markersize=2, color='g', label='selling signal', markevery=selling_signal,alpha=0.3)
    plt.title('actions signal')
    plt.legend()
    plt.xticks(range(-1, len(df.index)), df.index, rotation=90, ha='right')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(24))
    i_ep = df['episode'][-1]
    plt.savefig(image_dir + str(i_ep) + ".png",dpi=399)

if __name__ == "__main__":
    d = df.head(200)
    # peaks,bottoms = landmarks(d)
    attach_to(d,*landmarks(d))
    depict(d.close,*landmarks(d))