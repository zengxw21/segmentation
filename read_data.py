import gensim
import torch.nn as nn
import numpy as np
from collections import Counter
from torch.optim.lr_scheduler import *
from pathlib import Path
from typing import List,Dict
import os

def GetFilePath(filepath:str, filename:str)->str:
    files=os.listdir(filepath)
    for each in files:
        if each.startswith(f"{filename}"):
            return Path.cwd()/filepath/each
        
def GetFileList(filepath:str)->List[str]:
    files=os.listdir(filepath)
    ret=[]
    for each in files:
        if each.endswith(".txt") and not each.startswith("test"):
            ret.append(Path.cwd()/filepath/each)
    return ret

def Word2Id()->Dict:
    path=GetFileList("Dataset")
    word_id=Counter()
    for each in path:
        with open(each,encoding='utf-8',errors="ignore")as f:
            for line in f.readlines():
                sentence=line.strip().split()
                for word in sentence[1:]:
                    if word not in word_id.keys():
                        word_id[word]=len(word_id)
    return word_id

def Word2Vec(filename,word2id):
    path=GetFilePath("Dataset",filename)
    PreTrainModel=gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)
    word_vec=np.array(np.zeros([len(word2id)+1,PreTrainModel.vector_size]))
    for key in word2id:
        try:
            word_vec[word2id[key]]=PreTrainModel[key]
        except Exception:
            pass
    return word_vec

def GetCorpus(path,word2id,MaxLength=50):
    contents,labels=np.array([0]*MaxLength),np.array([])
    with open(path,encoding='utf-8',errors='ignore')as f:
        for line in f.readlines():
            sentence=line.strip().split()
            content=np.asarray([word2id.get(w,0)for w in sentence[1:]])[:MaxLength]
            padding=max(MaxLength-len(content),0)
            content=np.pad(content,((0,padding)),"constant",constant_values=0)
            labels=np.append(labels,int(sentence[0]))
            contents=np.vstack([contents,content])
    contents=np.delete(contents,0,axis=0)
    return contents,labels
