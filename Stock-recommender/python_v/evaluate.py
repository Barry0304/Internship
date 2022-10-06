from modules import *

VERSION = "opd(-1,20,1)_UfcS(epo50)"

user_data,stock_data,click_data,strategy_data\
    = load_data_from_pickle('./saves/'+str(VERSION)+'/data/data_pp.p') 

# evaluate_all(VERSION,click_data,stock_data,strategy_data)

stock_list=["0050","2330"]

show_from_evaluate_csv(VERSION,stock_list)
evaluate_stock(VERSION,stock_list,click_data,stock_data,strategy_data)