import pandas as pd
import numpy as np
import pickle
import json

__all__=['ke3','read_config']

def read_config(filename):
    with open(filename, encoding='utf-8') as f:
        config = json.load(f)
        f.close()
    return config

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
def tfidf(ws):
    vectorizer = CountVectorizer(stop_words=None,analyzer=lambda x: x)
    X = vectorizer.fit_transform(ws)
    transformer = TfidfTransformer(smooth_idf=True)
    Z = transformer.fit_transform(X)
    tfidf_df = pd.DataFrame(Z.toarray(),columns=vectorizer.get_feature_names_out ())
    return tfidf_df

from .textrank4zh import TextRank4Keyword, TextRank4Sentence
def text_rank(news,ws,pos):
    textrank_dic=[]
    for t,w,p in zip(news,ws,pos):
        tr4w = TextRank4Keyword()
        tr4w.analyze(text=t,myws=w,mypos=p, lower=True, window=2)
        result = tr4w.get_keywords(len(w), word_min_len=1)
        textrank_dic.append({item.word:item.weight for item in result})
    return textrank_dic

def gen_stopwords(ws,pos,pos_lib):
    stopwords = []
    for w,p in zip(ws,pos):
        if p in pos and pos_lib[p][1] == 0:
            stopwords.append(w)
    return stopwords

from .KeyExtractor.core import KeyExtractor
def key_bert(ws,pos,par_n_gram=1,par_top_n=30,stopwords=[],stopword_pos={},model='ckiplab/bert-base-chinese'):
    ke = KeyExtractor(embedding_method_or_model=model)
    keyword_score = []
    for w,p in zip(ws,pos):
        stopwords = stopwords+gen_stopwords(w,p,stopword_pos)
        keywords = ke.extract_keywords(
            w,stopwords=stopwords, n_gram=par_n_gram, top_n=par_top_n
        )
        result = []
        for k in keywords:
            result.append(["".join(k[1]) , k[2] , p[w.index("".join(k[1]))]])
        keyword_score.append(result)
        if len(keyword_score) %50 == 0:
            print(len(keyword_score)," news...done")
    return keyword_score

from scipy.stats import rankdata
from sklearn.preprocessing import normalize
def normalize_rank(news_data,tfidf_df,textrank_dic,keyword_score,score_weight,stopword_pos):
    columns=['newsID','word','pos','in_title','keybert_score','tfidf_score',"textrank_score",'keybert_rank','tfidf_rank',"textrank_rank",'total_rank']
    results=[]
    for (or_idx,row_data),idx in zip(news_data.iterrows(),range(len(news_data))):
        result_df=pd.DataFrame(columns=columns)
        if len(keyword_score[idx]) <2:
            results.append(result_df)
            continue
        result_df[['word','keybert_score','pos']] = [[key[0],key[1],key[2]] for key in keyword_score[idx]]
        result_df = result_df[result_df['pos'].apply(lambda x: (x not in stopword_pos )or (stopword_pos[x][1]>1))]
        
        result_df['in_title']=result_df['word'].apply(lambda x:10/len(result_df) if x in row_data['title'] else 0)
        result_df['tfidf_score']=result_df['word'].apply(lambda x: tfidf_df.at[or_idx,x] if x in tfidf_df.columns else 0)
        result_df['textrank_score']=result_df['word'].apply(lambda x: textrank_dic[idx][x] if x in textrank_dic[idx] else 0)
        
        result_df['keybert_rank']= normalize([result_df['keybert_score']])[0]
        result_df['tfidf_rank']= normalize([result_df['tfidf_score']])[0]
        result_df['textrank_rank']= normalize([result_df['textrank_score']])[0]

        result_df['total_rank']= rankdata(score_weight["title_weight"]*result_df['in_title']+score_weight["keybert_weight"]*result_df['keybert_rank']+score_weight["tfidf_weight"]*result_df['tfidf_rank']+score_weight["textrank_weight"]*result_df['textrank_rank'],method='min')
        result_df = result_df.sort_values(by=['total_rank'],ascending=False,ignore_index = True)
        result_df['newsID'] = row_data['newsID']
        results.append(result_df)
    return results

def save_data(file_name,r):
    with open(file_name, "wb") as fp:
        pickle.dump(r, fp)

def ke3(news_data,ws,pos,config,stopword_pos={},target=0):
    '''
        target表倒數n筆為目標
    '''
    print("start analysis:")
    try:
        tfidf_df=tfidf(ws)
    except:
        news_data = news_data[:config["tfidf"]["MAX_news_num"]]
        ws = ws[:config["tfidf"]["MAX_news_num"]]
        pos = pos[:config["tfidf"]["MAX_news_num"]]
        tfidf_df=tfidf(ws)
    print("tfidf...done")

    if target:
        news_data = news_data[-target:]
        ws = ws[-target:]
        pos = pos[-target:]

    textrank_dic=text_rank(news_data['content'],ws,pos)
    print("textrank...done")

    print("keyword_score...start")
    keyword_score = key_bert(ws,pos,\
        par_n_gram=1,\
        par_top_n=config['keybert']['par_top_n'],
        stopwords=config['keybert']['stopwords'],
        stopword_pos=stopword_pos,\
        model = config['keybert']['model']
    )
    print("keyword_score...done")

    results = normalize_rank(news_data,tfidf_df,textrank_dic,keyword_score,config['score'],stopword_pos)
    
    save_data("./data/news_keyword_dfl.p",results)
    return results
