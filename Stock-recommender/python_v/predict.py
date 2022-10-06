from modules import *

VERSION = "opd(-1,20,1)_UfcS(epo50)"

features, targets_values, users, stocks, clicks,  \
    stockID_map,strategyID_map,userID_map,mkID_map,stocksid2idx,usersid2idx \
        = load_data_from_pickle('./saves/'+str(VERSION)+'/data/data_fm.p') 

user_data,stock_data,click_data,strategy_data\
    = load_data_from_pickle('./saves/'+str(VERSION)+'/data/data_pp.p') 

stock_matrics = load_from_npy('./saves/'+str(VERSION)+'/feature_matrix/stock_matrics.npy')
# users_matrics = load_from_npy('./saves/'+str(VERSION)+'/feature_matrix/users_matrics.npy')

def sID_filter(x):
    if x.isdigit() and len(x)==4 :
        return True
    else:
        return False

# generate_result_df(stockID_map,mkID_map,stock_matrics,stocksid2idx,stocks,sID_filter,VERSION)
# evaluate_all(VERSION,click_data,stock_data,strategy_data)

stock_list=['2330','2885','2884','2303','2412']

show_from_result_csv(VERSION,stock_list)
show_from_evaluate_csv(VERSION,stock_list)

results=recommend_same_type_stock(stock_list, 20,stockID_map,mkID_map,stock_matrics,stocksid2idx,stocks,sID_filter)
for i in results:
    print("您選擇的股票為：",i[0],i[1])
    print("以下是给您的推薦：")
    for j in i[2]:
        print(j)

evaluate_stock(VERSION,stock_list,click_data,stock_data,strategy_data)

