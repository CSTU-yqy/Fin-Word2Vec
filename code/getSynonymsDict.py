from re import match
import joblib
import numpy as np
import pandas as pd
import torch
from bidict import bidict
from tqdm import tqdm

if __name__ == "__main__":
    state = ""
    word_vec = joblib.load("/root/autodl-tmp/PAPER/DATA/lexicon/wordvec/wordvec%s.pkl"%state)
    word2idx = bidict(joblib.load("/root/autodl-tmp/PAPER/DATA/lexicon/wordvec/word2idx%s.pkl"%state))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word_vec = torch.tensor(word_vec)
    word_vec = word_vec.to(device)

    def get_similar_word(target_word,k,word_vec):
        global word2idx
        idx = word2idx[target_word]
        x = word_vec[idx]
        cos = torch.matmul(word_vec,x) / (torch.sum(word_vec * word_vec,dim = 1) * torch.sum(x * x) + 1e-9).sqrt()
        _,topk = torch.topk(cos,k = k + 1)
        topk = topk.cpu().numpy()
        return {
            target_word:[word2idx.inverse[j] for j in topk[1:]]
        }
    synonyms_dict = dict()
    for word in tqdm(word2idx.keys()):
        tk = list(get_similar_word(word,20,word_vec).values())[0]
        if word not in synonyms_dict.keys():
            synonyms_dict[word] = tk
    joblib.dump(synonyms_dict,"/root/autodl-tmp/PAPER/DATA/RAW_DATA/important/synonyms_dict.pkl")
        
    