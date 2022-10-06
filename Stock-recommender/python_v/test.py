import pandas as pd
import os
def load_data_from_file(col,file_dir,drop_subset):
    df = pd.DataFrame(columns=col)
    for file in os.listdir(file_dir):
        df = df.append(pd.read_csv(file_dir+'/'+file, header=0,names = col))
    df=df.drop_duplicates(subset=drop_subset, keep='last')
    return df

click_col = ['userID_or','mkID_stockID','clicks_or']
click_data = load_data_from_file(click_col,'./data/click',click_col[0:2])
click_data.to_csv("test.csv", index=False)
