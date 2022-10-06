from modules import *
import os

VERSION = "opd(-1,20,1)_UfcS(epo3)"
PATH = './saves/'+str(VERSION)

if not os.path.exists(PATH):
    os.mkdir(PATH)
    folders=["data","model","loss_fig","feature_matrix"]
    for fn in folders:
        os.mkdir(PATH+"/"+fn)

'''
data
'''
stock_per_strategy,strategy_data,click_data \
    = load_data()

'''
['non_drop',[]]
['slapdash_drop',[]]
['rigorous_drop',[lb_n,ub_1clk,ub_uatt,ub_utc]]
['optimaize_drop',[ua_lb(set -1 for auto),lb_n,ub_nstd]]
'''
method=['optimaize_drop',[-1,20,1]]

def sID_filter(x):
    if x.isdigit() and 0<int(x)<10000 :
        return True
    else:
        return False

user_data,stock_data,click_data,strategy_data,\
    features, targets_values, users, stocks, clicks,  \
        stockID_map,strategyID_map,userID_map,mkID_map,stocksid2idx,usersid2idx \
            = data_pp(stock_per_strategy,strategy_data,click_data,method,VERSION)#,sID_filter)


# user_data,stock_data,click_data,strategy_data\
#     = load_data_from_pickle('./saves/'+str(VERSION)+'/data/data_pp.p')
# features, targets_values, users, stocks, clicks,  \
#         stockID_map,strategyID_map,userID_map,mkID_map,stocksid2idx,usersid2idx \
#             = load_data_from_pickle('./saves/'+str(VERSION)+'/data/data_fm.p')
 

'''
model
'''
mv_net=train(features, targets_values,strategyID_map,VERSION,3)
save_loss_map(mv_net,'train',VERSION)
save_loss_map(mv_net,'test',VERSION)
save_loss_map(mv_net,'epo_train',VERSION)
save_loss_map(mv_net,'epo_test',VERSION)

# mv_net=load_from_h5('./saves/'+str(VERSION)+'/model/model.h5',features,strategyID_map)


'''
predict
'''
stock_matrics=create_stocks_matrics(mv_net,stocks,VERSION)
users_matrics=create_users_matrics(mv_net,users,VERSION)
# "要跑很久"inference_matrics=create_inference_matrics(mv_net,users_matrics,stock_matrics,VERSION)
# stock_matrics = load_from_npy('./saves/'+str(VERSION)+'/feature_matrix/stock_matrics.npy')
# users_matrics = load_from_npy('./saves/'+str(VERSION)+'/feature_matrix/users_matrics.npy')

generate_result_df(20,stockID_map,mkID_map,stock_matrics,stocksid2idx,stocks,sID_filter,VERSION)

evaluate_all(VERSION,click_data,stock_data,strategy_data)
