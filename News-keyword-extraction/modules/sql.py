import pyodbc
import pandas as pd
import json

__all__=['read_config','load_from_sql','kdfl_2_sql']

def read_config(filename):
    with open(filename, encoding='utf-8') as f:
        config = json.load(f)
        f.close()
    return config

def connect_to_sql(config):
    cnxn = pyodbc.connect(
        'DRIVER={'+config['driver']+'};' \
        +'SERVER='+config['server']+','+config['port']+';' \
        +'DATABASE='+config['database']+';' \
        +'UID='+config['username']+';' \
        +'PWD='+config['password']
    )
    cursor = cnxn.cursor()
    return cnxn,cursor

def close_connect(cnxn,cursor):
    cnxn.commit()
    cursor.close()
    cnxn.close()

def load_data(cnxn,table_name):
    df = pd.read_sql("SELECT * FROM "+table_name, cnxn)
    return df

def load_from_sql(config,table_name):
    cnxn,cursor=connect_to_sql(config)
    df = load_data(cnxn,table_name)
    df.to_csv('data/news_data.csv', index=False)
    close_connect(cnxn,cursor)
    return df    

def kdfl_2_sql(dfl,table_name,config):
    cnxn,cursor=connect_to_sql(config)
    insert="""
        INSERT INTO [keyword_data]([newsID],
            [word],
            [pos],
            [in_title],
            [keybert_score],
            [tfidf_score],
            [textrank_score],
            [keybert_rank],
            [tfidf_rank],
            [textrank_rank],
            [total_rank]
        )
        VALUES (?, ?, ?, ?, ?, ?,?,?,?,?,? )
    """
    for news_keyword in dfl:
        for i,r in news_keyword.iterrows():
            cursor.execute(insert,*r.to_list())
    close_connect(cnxn,cursor)
