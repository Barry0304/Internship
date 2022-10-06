from modules import *

sql_config = read_config('config/sql_config.json')

news_data=load_from_sql(sql_config,'[dbo].[SS_News]')

model_config = read_config('config/model_config.json')

news_data = load_data("data/SS_News.csv")

news_data=news_data[:20]

news_data,ws,pos=ws(news_data,"data/news_nlp.data",model_config)

extract_config = read_config('config/extract_config.json')

stopwords = read_config('config/pos_arg.json')

keyword_dfl=ke3(news_data,ws,pos,extract_config,stopwords,target=10)

kdfl_2_sql(keyword_dfl,'keyword_data',sql_config)