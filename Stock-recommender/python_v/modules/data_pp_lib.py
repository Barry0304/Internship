import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import os

__all__=['load_data','data_pp','load_data_from_pickle']

def load_data_from_file(col,file_dir,drop_subset):
    df = pd.DataFrame(columns=col)
    for file in os.listdir(file_dir):
        df = df.append(pd.read_csv(file_dir+'/'+file, header=0,names = col))
    df=df.drop_duplicates(subset=drop_subset, keep='last')
    return df

def non_drop(click_data):
    return click_data.reset_index(drop=True)

def slapdash_drop(click_data):
    click_data=click_data.groupby('userID_or').filter(lambda x: len(x) > 1)
    return click_data.reset_index(drop=True)

def rigorous_drop(lb_n,ub_1clk,ub_uatt,ub_utc,click_data):
    #drop outliers
    n=0
    while(n!=len(click_data)):
        n=len(click_data)
        #drop (click_n) outliers    
        click_data=click_data[click_data['clicks_or'].apply(lambda x: x < ub_1clk)]
        #drop (stock attention) outliers
        click_data=click_data.groupby('stockID_or').filter(lambda x: len(x) > lb_n)
        #drop (user attention/user total_click) outliers
        click_data=click_data.groupby('userID_or').filter(lambda x: lb_n < len(x) < ub_uatt)
        click_data=click_data.groupby('userID_or').filter(lambda x: x['clicks_or'].sum()<ub_utc)
    return click_data.reset_index(drop=True)
    
def mean_on_n_std(n,data):
    return data.mean()+(n*data.std())

def optimaize_drop(ua_lb,sa_lb,ub_nstd,click_data):   
    #drop outliers
    if ua_lb == -1:
        ua_lb=click_data.groupby('userID_or').size().describe()['50%']
    if sa_lb == -1:
        sa_lb=click_data.groupby('stockID_or').size().describe()['75%']
    c_ub=mean_on_n_std(ub_nstd,click_data['clicks_or'])
    ua_ub=mean_on_n_std(ub_nstd,click_data.groupby('userID_or').size())
    utc_ub=mean_on_n_std(ub_nstd,click_data.groupby('userID_or')['clicks_or'].sum())
    n=0
    while(n!=len(click_data)):
        n=len(click_data)
        #drop (click_n) outliers    
        click_data=click_data[click_data['clicks_or'].apply(lambda x: x < c_ub)]
        #drop (stock attention) outliers
        click_data=click_data.groupby('stockID_or').filter(lambda x: (sa_lb < len(x)))
        #drop (user attention/user total_click) outliers
        click_data=click_data.groupby('userID_or').filter(lambda x: ua_lb < len(x) < ua_ub)
        click_data=click_data.groupby('userID_or').filter(lambda x: x['clicks_or'].sum()<utc_ub)

    return click_data.reset_index(drop=True)

def drop_wrong_click_data(sID_filter,click_data):
    #drop test account('VAC.....',"...~Y")
    click_data=click_data[click_data['userID_or'].apply(lambda x: (len(x) == 18) and (x[-1]=='Y'))]
    temp=click_data.copy()
    temp[['mkID_or','stockID_or']] = [[id[0],id[-1]] for id in click_data['mkID_stockID'].str.split('_')]
    click_data=temp
    
    #drop empty stockID
    click_data=click_data[~click_data['stockID_or'].isin(['','1','100000'])]
    click_data=click_data[click_data['mkID_or'].apply(lambda x: x in ['1','2','3','4','5','6'])]
    click_data=click_data[click_data['stockID_or'].apply(lambda x: sID_filter(x))]

     #drop outlier
    uatt_ub=click_data.groupby('userID_or').size().describe([.95]).at['95%']
    utotal_ub=click_data.groupby('userID_or')['clicks_or'].sum().describe([.95]).at['95%']    
    click_data=click_data.groupby('userID_or').filter(lambda x: len(x) < uatt_ub and x['clicks_or'].sum()<utotal_ub)

    return click_data.reset_index(drop=True)

def click_data_pp(method,sID_filter,click_data):
    click_data=drop_wrong_click_data(sID_filter,click_data)
    click_data['clicks_or'] =click_data['clicks_or'].astype(int)
    fun_dict={
        'non_drop' : non_drop,
        'slapdash_drop' : slapdash_drop,
        'rigorous_drop' : rigorous_drop,
        'optimaize_drop' : optimaize_drop
    }

    click_data=fun_dict[method[0]](*method[1],click_data)

    return click_data

def generate_lost_user_data(click_data):
    user_data = pd.DataFrame()
    user_data["userID_or"]=click_data['userID_or'].unique()
    # user_data["Gender"] = np.zeros(user_data.shape[0], dtype = int)
    # user_data["Age"] = np.zeros(user_data.shape[0], dtype = int)
    # user_data["JobID"] = np.zeros(user_data.shape[0], dtype = int)
    # user_data["Locate"] = np.zeros(user_data.shape[0], dtype = int)

    return user_data

def user_data_pp(click_data):
    user_data = generate_lost_user_data(click_data)

    return user_data

def generate_lost_stock_data(stock_data,click_data):
    s_IDL=stock_data['stockID_or'].unique()
    c_IDL=click_data.drop_duplicates(subset=['stockID_or'])[['mkID_or','stockID_or']]
    c_IDL=c_IDL[c_IDL['stockID_or'].apply(lambda x : x not in s_IDL)]
    c_IDL['strategyList_or']=c_IDL.apply(lambda x: [], axis=1)
    stock_data=stock_data.append(c_IDL,ignore_index=True)
    
    return stock_data

def stock_data_pp(stock_per_strategy,click_data):
    gp=stock_per_strategy.groupby(['stockID_or'],as_index=False)
    stock_data = [[val['mkID_or'].mode().item(),key,np.unique(val['strategyID_or'])] for key, val in gp]
    stock_col = ['mkID_or','stockID_or','strategyList_or']
    stock_data=pd.DataFrame(stock_data,columns=stock_col)
    #因stock_per_strategy有缺失
    stock_data=generate_lost_stock_data(stock_data,click_data)
    #stock strategy數
    stock_data['strategy_n']=stock_data["strategyList_or"].apply(lambda x: len(x))
    stock_data=stock_data.astype({'mkID_or': 'str'})

    return stock_data

def cut2scale(x,lable_n):
    re = pd.qcut(x,lable_n,labels=False, duplicates='drop')
    re=re.fillna(0)
    re= ((re+1)*lable_n/((re.max()+1))).round(0)
    return re

def click_data_fe(click_data):
    # lable_n=5
    # click_data['clicks'] = click_data.groupby(['userID_or'])['clicks_or'].transform(lambda x:cut2scale(x,lable_n))
    # click_data['clicks'] =click_data['clicks'].astype(int)
    click_data['clicks'] = click_data.groupby(['userID_or'])['clicks_or'].transform(lambda x: (x - x.mean()) / x.std())
    return click_data

def user_data_fe(user_data,click_data):
    temp = click_data.groupby('userID_or').size().to_dict()
    user_data['u_attentions'] = user_data['userID_or'].map(temp).fillna(0).astype(int)

    temp = click_data.groupby('userID_or')['clicks_or'].sum().to_dict()
    user_data['u_total_clicks'] = user_data['userID_or'].map(temp).fillna(0).astype(int)

    user_data=user_data[~(user_data['u_attentions']==0)].reset_index(drop=True)

    return user_data

def stock_data_fe(stock_data,click_data):
    temp= click_data.groupby('stockID_or').size().to_dict()
    stock_data['s_attentions'] = stock_data['stockID_or'].map(temp).fillna(0).astype(int)

    temp = click_data.groupby('stockID_or')['clicks_or'].sum().to_dict()
    stock_data['s_total_clicks'] = stock_data['stockID_or'].map(temp).fillna(0).astype(int)

    stock_data=stock_data[~(stock_data['s_attentions']==0)].reset_index(drop=True)

    return stock_data

def le_map(df,or_col,new_col):
    le=LabelEncoder()
    df[new_col] = le.fit_transform(df[or_col])
    IDmap={df[or_col][i]:df[new_col][i] for i in df.index}
    
    return df,IDmap

def data_le(strategy_data,click_data,user_data,stock_data):
    strategy_data,strategyID_map = le_map(strategy_data,'strategyID_or',"strategyID_le")
    user_data,userID_map = le_map(user_data,'userID_or','userID_le')
    stock_data,stockID_map = le_map(stock_data,"stockID_or","stockID_le")
    stock_data,mkID_map = le_map(stock_data,"mkID_or","mkID_le")
    stock_data['strategyList_le']=stock_data['strategyList_or'].transform(lambda x:[strategyID_map[ID] for ID in x])
    click_data['userID_le'] = click_data['userID_or'].map(userID_map)
    click_data['stockID_le'] = click_data['stockID_or'].map(stockID_map)

    return strategy_data,user_data,stock_data,click_data,strategyID_map,userID_map,stockID_map,mkID_map

def re_scaler(data):
    scaler=MinMaxScaler(feature_range=(0,1))
    return scaler.fit_transform(data)

def data_format(strategyID_map,click_data, user_data, stock_data):
    #strategyList padding
    users = user_data[['userID_le','u_attentions','u_total_clicks']]
    stocks= stock_data[['stockID_le','mkID_le','s_attentions','s_total_clicks','strategyList_le']]
    clicks = click_data[['userID_le','stockID_le', 'clicks']]
    magic=max(strategyID_map.values())
    strategyID_map['0']=magic
    for i in range(len(stocks)):
        stocks.at[i,'strategyList_le'] = stocks.at[i,'strategyList_le']+[magic]*(40 - len(stocks.at[i,'strategyList_le']))
    data = pd.merge(pd.merge(clicks, users), stocks)
    target_fields = ['clicks']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]
    features = features_pd.values
    targets_values = targets_pd.values
    stocksid2idx = {val[0]:i for i, val in enumerate(stocks.values)}
    usersid2idx = {val[0]:i for i, val in enumerate(users.values)}
    return features,targets_values,clicks, users, stocks,stocksid2idx,usersid2idx

def load_data(stock_per_strategy_file='./data/stock',strategy_data_file='./data/strategy',click_data_file='./data/click'):
    stock_col = ['mkID_or','stockID_or','strategyID_or','strategyCn']
    strategy_col = ['strategyID_or','strategyCN']
    click_col = ['userID_or','mkID_stockID','clicks_or']
    #load file
    stock_per_strategy = load_data_from_file(stock_col,stock_per_strategy_file,stock_col[0:3])
    strategy_data = load_data_from_file(strategy_col,strategy_data_file,strategy_col[0:2])
    click_data = load_data_from_file(click_col,click_data_file,click_col[0:2])
    return stock_per_strategy,strategy_data,click_data

def data_pp(stock_per_strategy,strategy_data,click_data,method,VERSION='test',sID_filter=lambda x:True):
    #pp
    click_data = click_data_pp(method,sID_filter,click_data)
    user_data = user_data_pp(click_data)
    stock_data = stock_data_pp(stock_per_strategy,click_data)
    #feature engineering
    click_data = click_data_fe(click_data)
    user_data = user_data_fe(user_data,click_data)
    stock_data = stock_data_fe(stock_data,click_data)
    #le
    strategy_data,user_data,stock_data,click_data,\
        strategyID_map,userID_map,stockID_map,mkID_map\
            = data_le(strategy_data,click_data,user_data,stock_data)
    
    f = open('./saves/'+str(VERSION)+'/data/data.log', "a")
    print(click_data[['clicks','clicks_or']].describe(), file = f)
    print(user_data[['u_attentions','u_total_clicks']].describe(), file = f)
    print(stock_data[['strategy_n','s_attentions','s_total_clicks']].describe(), file = f)
    f.close()
    
    #MinMaxScaler
    stock_data[['s_attentions','s_total_clicks']] = re_scaler(stock_data[['s_attentions','s_total_clicks']])
    user_data[['u_attentions','u_total_clicks']] = re_scaler(user_data[['u_attentions','u_total_clicks']])

    #format_for_model
    features,targets_values,clicks, users, stocks,stocksid2idx,usersid2idx= data_format(strategyID_map,click_data, user_data, stock_data)
    #save data
    pickle.dump((user_data,stock_data,click_data,strategy_data), open('./saves/'+str(VERSION)+'/data/data_pp.p', 'wb'))
    pickle.dump((features, targets_values, users, stocks, clicks, stockID_map,strategyID_map,userID_map,mkID_map,stocksid2idx,usersid2idx), open('./saves/'+str(VERSION)+'/data/data_fm.p', 'wb'))

    return user_data,stock_data,click_data,strategy_data,\
              features, targets_values, users, stocks, clicks,\
                stockID_map,strategyID_map,userID_map,mkID_map,stocksid2idx,usersid2idx

def load_data_from_pickle(filename):
    return pickle.load(open(filename, mode='rb'))