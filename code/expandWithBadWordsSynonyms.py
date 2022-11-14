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
    bad_word_seed = list(set(open("/root/autodl-tmp/PAPER/DATA/lexicon/bad_news_dictionary/raw_data/bad_words.txt").read().split("\n")))
    
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
    
    bad_word_dictionary = {}
    for bw in tqdm(bad_word_seed):
        if bw in word2idx.keys():
            bad_word_dictionary.update(get_similar_word(bw,10,word_vec))
    bad_word_list = []
    with open("/root/autodl-tmp/PAPER/DATA/lexicon/bad_news_dictionary/use_fin_word2vec/bad_words_expand%s.txt"%state,"w") as f:
        for k,v in bad_word_dictionary.items():
            f.write( "\n" + k + "\n")
            bad_word_list += v
            for w in v:
                f.write(w + "\n")
        f.close()
    bad_word_list = sorted(list(set(bad_word_list)))
    with open("/root/autodl-tmp/PAPER/DATA/lexicon/bad_news_dictionary/use_fin_word2vec/bad_words_dictionary%s.txt"%state,"w") as f:
        for b in bad_word_list:
            f.write( "\n" + b)
        f.close()
    





    


