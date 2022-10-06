import pandas as pd
import numpy as np
import pickle
from .ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import torch

__all__=['load_data','ws']

def load_data(file_name):
    news_col = ['newsID','date','source','mkID','mk','title','content','updateTime','display']
    news_data=pd.read_csv(file_name, header=0,names = news_col)
    return news_data

def preprocess(news_data):
    news_data['title'] = news_data['title'].apply(lambda x : x.replace(" ", "").replace("\t", "").replace("　", ""))
    news_data['content'] = news_data['content'].apply(lambda x : x.replace(" ", "").replace("\t", "").replace("　", ""))
    return news_data

def load_driver(model_config):
    f = open('excute.config', "w")
    print("torch_version : " + str(torch.__version__), file = f)
    print("cuda_version : " + str(torch.version.cuda), file = f)
    print("cuda_is_available : " + str(torch.cuda.is_available()), file = f)
    print("device : " + str(torch.cuda.get_device_name()), file = f)
    f.close()

    ws_driver = CkipWordSegmenter(model_name=model_config["ws_model"],device=model_config["device"])
    pos_driver = CkipPosTagger(model_name=model_config["pos_model"],device=model_config["device"])
    ner_driver = CkipNerChunker(model_name=model_config["ner_model"],device=model_config["device"])
    return ws_driver,pos_driver,ner_driver

def main_process(text,model_config,ws_driver,pos_driver,ner_driver):
    all_ws = ws_driver(text,batch_size=model_config['batch_size'])
    all_pos = pos_driver(all_ws,batch_size=model_config['batch_size'])
    all_ner = ner_driver(text,batch_size=model_config['batch_size'])
    return all_ws,all_pos,all_ner

def save_data(save_file,all_ws,all_pos,all_ner):
    with open(save_file, "wb") as fp:
        pickle.dump((all_ws,all_pos,all_ner), fp)

def ws(news_data,save_file,model_config):
    news_data = preprocess(news_data)
    ws_driver,pos_driver,ner_driver = load_driver(model_config)
    text = list(news_data['title']+'。'+news_data['content'])
    all_ws,all_pos,all_ner = main_process(text,model_config,ws_driver,pos_driver,ner_driver)
    save_data(save_file,all_ws,all_pos,all_ner)
    return(news_data,all_ws,all_pos)
    
