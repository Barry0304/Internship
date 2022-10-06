import pandas as pd

__all__=['evaluate_stock','evaluate_all','show_from_evaluate_csv']

def same_strategy(sID1,sID2,stock_data,strategy_data):
    set_x=set(*stock_data[stock_data['stockID_or']==sID1]['strategyList_or'])
    set_y=set(*stock_data[stock_data['stockID_or']==sID2]['strategyList_or'])
    result_df = pd.DataFrame((set_x & set_y),columns=['strategyID_or'])
    result_df=result_df.merge(strategy_data[['strategyID_or','strategyCN']], how='inner', on='strategyID_or')
    return result_df

def clk2clk_rate(sID1,sID2,click_gp):
    set_x=set(click_gp.get_group(sID1)['userID_or'])
    set_y=set(click_gp.get_group(sID2)['userID_or'])
    a2b=len(set_x & set_y)/len(set_x)
    b2a=len(set_x & set_y)/len(set_y)
    return a2b,b2a

def evaluate_stock(VERSION,stock_list,click_data,stock_data,strategy_data):
    result_type={'MarketNo': 'str', 'StockID': 'str', 'RelatedMarketNo': 'str', 'RelatedStockID': 'str'}
    result_df=pd.read_csv('./saves/'+str(VERSION)+'/result.csv', header=0, dtype=result_type)
    click_gp = click_data.groupby("stockID_or")
    for sID in stock_list:
        for index, row in result_df[result_df['StockID']==sID].iterrows():
            t_ID=row["MarketNo"]+'-'+row["StockID"]
            r_ID=row['RelatedMarketNo']+'-'+row['RelatedStockID']
            print("對"+t_ID+'推薦'+r_ID)
            print("可能原因:\n\t相同策略:")
            ss_df=same_strategy(row["StockID"],row["RelatedStockID"],stock_data,strategy_data)
            for i,j in zip(ss_df['strategyID_or'],ss_df['strategyCN']):
                print("\t\t"+str(i)+":"+j)
            t2r,r2t=clk2clk_rate(row["StockID"],row["RelatedStockID"],click_gp)
            print("\n\t點擊"+t_ID+"的使用者有"+'{:.3%}'.format(t2r)+"點擊了"+r_ID)
            print("\t點擊"+r_ID+"的使用者有"+'{:.3%}'.format(r2t)+"點擊了"+t_ID+"\n")

def evaluate_all(VERSION,click_data,stock_data,strategy_data):
    result_type={'MarketNo': 'str', 'StockID': 'str', 'RelatedMarketNo': 'str', 'RelatedStockID': 'str'}
    result_df=pd.read_csv('./saves/'+str(VERSION)+'/result.csv', header=0, dtype=result_type)
    evaluate_df_col=['t_mID','t_sID','r_mID','r_sID','same_strategy','t2r','r2t']
    click_gp = click_data.groupby("stockID_or")
    data=[]
    for index, row in result_df.iterrows():
        temp=[]    
        #target_id
        temp.extend([row["MarketNo"],row["StockID"]])
        #recommend_id
        temp.extend([row["RelatedMarketNo"],row["RelatedStockID"]])
        #same_strategy
        ss_df=same_strategy(row["StockID"],row["RelatedStockID"],stock_data,strategy_data)
        temp.append([[i,j] for i,j in zip(ss_df['strategyID_or'],ss_df['strategyCN'])])
        #t2r    
        temp.extend(clk2clk_rate(row["StockID"],row["RelatedStockID"],click_gp))

        data.append(temp)
    evaluate_df=pd.DataFrame(data,columns=evaluate_df_col)
    evaluate_df.to_csv("./saves/"+VERSION+"/evaluate.csv", index=False)

def show_from_evaluate_csv(VERSION,stock_list=-1):
    evaluate_type={'t_mID': 'str', 't_sID': 'str','r_mID': 'str', 'r_sID': 'str'}
    evaluate_df=pd.read_csv('./saves/'+str(VERSION)+'/evaluate.csv', header=0, dtype=evaluate_type)
    if stock_list == -1:
        stock_list = evaluate_df['t_sID'].unique()
    for sID in stock_list:
        for index, row in evaluate_df[evaluate_df['t_sID']==sID].iterrows():
            t_ID=row['t_mID']+'-'+row['t_sID']
            r_ID=row['r_mID']+'-'+row['r_sID']
            print("對"+t_ID+'推薦'+r_ID)
            print("可能原因:\n\t相同策略:")
            for ss in eval(row['same_strategy']):
                print("\t\t"+str(ss[0])+":"+ss[1])
            print("\n\t點擊"+t_ID+"的使用者有"+'{:.3%}'.format(row['t2r'])+"點擊了"+r_ID)
            print("\t點擊"+r_ID+"的使用者有"+'{:.3%}'.format(row['r2t'])+"點擊了"+t_ID+"\n")
