import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

__all__=['create_stocks_matrics','create_users_matrics','load_from_npy','recommend_same_type_stock','generate_result_df','show_from_result_csv']

def create_stocks_matrics(mv_net,stocks,VERSION):
    stock_layer_model = keras.models.Model(inputs=[mv_net.model.input[3], mv_net.model.input[4], mv_net.model.input[5], mv_net.model.input[6], mv_net.model.input[7]], 
                                 outputs=mv_net.model.get_layer("stock_combine_layer_flat").output)
    stock_matrics = []

    for item in stocks.values:
        strategies = np.zeros([1, 40])
        strategies[0] = item.take(4)
        stock_combine_layer_flat_val = stock_layer_model([np.reshape(item.take(0), [1, 1]),
                                                        np.reshape(item.take(1), [1, 1]),
                                                        np.reshape(item.take(2), [1, 1]),
                                                        np.reshape(item.take(3), [1, 1]),
                                                        strategies])  
        stock_matrics.append(stock_combine_layer_flat_val)
    stock_matrics = (np.array(stock_matrics).reshape(-1, 400))
    np.save('./saves/'+str(VERSION)+'/feature_matrix/stock_matrics.npy', stock_matrics)
    return stock_matrics

def create_users_matrics(mv_net,users,VERSION):
    user_layer_model = keras.models.Model(inputs=[mv_net.model.input[0], mv_net.model.input[1], mv_net.model.input[2]], 
                                    outputs=mv_net.model.get_layer("user_combine_layer_flat").output)
    users_matrics = []

    for item in users.values:

        user_combine_layer_flat_val = user_layer_model([np.reshape(item.take(0), [1, 1]), 
                                                        np.reshape(item.take(1), [1, 1]), 
                                                        np.reshape(item.take(2), [1, 1])])
        users_matrics.append(user_combine_layer_flat_val)

    users_matrics = (np.array(users_matrics).reshape(-1, 400))
    np.save('./saves/'+str(VERSION)+'/feature_matrix/users_matrics.npy', users_matrics)
    return users_matrics

def create_inference_matrics(mv_net,users_matrics,stock_matrics,VERSION):
    inference_layer_model = keras.models.Model(inputs=mv_net.model.get_layer("inference_layer").input, 
                                    outputs=mv_net.model.get_layer("inference").output)
    inference_matrics = []
    for u_item in users_matrics:
        temp=[]
        for s_item in stock_matrics:
            inference_layer_val = inference_layer_model([np.reshape(u_item, [1, 400]),
                                                         np.reshape(s_item, [1, 400])])
            temp.append(inference_layer_val)
        inference_matrics.append(temp)
    inference_matrics = (np.array(inference_matrics).reshape(-1, len(stock_matrics)))
    np.save('./saves/'+str(VERSION)+'/feature_matrix/inference_matrics.npy', inference_matrics)
    return inference_matrics

def load_from_npy(filename):
    return np.load(filename)

def dict_values2key(Dict,value):
    return str(list(Dict.keys())[list(Dict.values()).index(value)])

def recommend_same_type_stock(predict_stock_list,top_k,stockID_map,mkID_map,stock_matrics,stocksid2idx,stocks,sID_filter = lambda x:True):
    results=[]
    for stockID in predict_stock_list:
        stockID_le=stockID_map[stockID]
        mkID = dict_values2key(mkID_map,int(stocks[stocks['stockID_le']==stockID_le]['mkID_le'].values))
        sim = cosine_similarity((stock_matrics[stocksid2idx[stockID_le]]).reshape([1, 400]),stock_matrics)
        sim = np.squeeze(sim)
        sim[stocksid2idx[stockID_le]] = -1
        temp=[]
        for idx in np.argsort(-sim):
            if len(temp) == top_k or sim[idx] < 0.5:
                break
            re_stockID = dict_values2key(stockID_map,stocks.at[idx,'stockID_le'])
            if (not sID_filter(re_stockID)):
                continue
            else:
                re_mkID = dict_values2key(mkID_map,stocks.at[idx,'mkID_le'])
                temp.append([re_mkID,re_stockID,sim[idx]])
        results.append([mkID,stockID,temp])
    return results

def generate_result_df(top_k,stockID_map,mkID_map,stock_matrics,stocksid2idx,stocks,sID_filter,VERSION):
    stock_list=list(filter(lambda x : sID_filter(x),stockID_map.keys()))
    results=recommend_same_type_stock(stock_list,top_k,stockID_map,mkID_map,stock_matrics,stocksid2idx,stocks,sID_filter)
    temp=[]
    for i in results:
        for j in i[2]:
            temp.append([i[0],i[1],VERSION,VERSION,j[0],j[1],j[2]])
                
    result_df_col=['MarketNo','StockID','SourceID','SourceName','RelatedMarketNo','RelatedStockID','RelatedStockProbability']
    result_df=pd.DataFrame(temp,columns=result_df_col)

    result_df.to_csv("./saves/"+VERSION+"/result.csv", index=False)

def show_from_result_csv(VERSION,stock_list=-1):
    result_type={'MarketNo': 'str', 'StockID': 'str', 'RelatedMarketNo': 'str', 'RelatedStockID': 'str'}
    result_df=pd.read_csv('./saves/'+str(VERSION)+'/result.csv', header=0, dtype=result_type)
    if stock_list == -1:
        stock_list = result_df['StockID'].unique()
    for sID in stock_list:
        first=True
        for index, row in result_df[result_df['StockID']==sID].iterrows():
            if first:
                t_ID=row['MarketNo']+'-'+row['StockID']
                print("選擇的股票為"+t_ID)
            r_ID=row['RelatedMarketNo']+'-'+row['RelatedStockID']
            print("以下是给您的推薦：")
            print('\t'+r_ID+'有{:.3%}'.format(row['RelatedStockProbability']))
